"""Paper-faithful BSPGA engine.

Implements the non-revisiting genetic algorithm of Su, Guo, Tian, and Zhang
(Information Sciences 512, 2020) using the paper's binary chromosome, novel
binary-space-partition tree, BSP-tree learning, uniform crossover, bit-flip
mutation, and elitist environmental selection.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .protocol import (
    BaseEngine,
    CandidateRecord,
    CapabilityProfile,
    EngineConfig,
    EngineState,
    OptimizationResult,
    ProblemSpec,
)


@dataclass
class _BSPNode:
    """A paper-style BSP-tree node.

    ``level`` is one-based, matching Algorithms 2 and 3 in the paper.  A leaf
    at level k is split using chromosome bit k.  When a leaf is split, its
    historical solution remains in the internal node and copies are placed in
    the two children, which preserves the tree's search history.
    """

    bits: np.ndarray
    position: np.ndarray
    fitness: float
    level: int
    left: "_BSPNode | None" = None
    right: "_BSPNode | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class BSPGAEngine(BaseEngine):
    """Non-revisiting GA based on the novel binary space partition tree."""

    algorithm_id = "bspga"
    algorithm_name = (
        "Non-Revisiting Genetic Algorithm Based on a Novel Binary Space "
        "Partition Tree"
    )
    family = "evolutionary"
    _REFERENCE = {
        "doi": "10.1016/j.ins.2019.10.016",
        "authors": "Y. Su, N. Guo, Y. Tian, and X. Zhang",
        "title": "A non-revisiting genetic algorithm based on a novel binary space partition tree",
        "journal": "Information Sciences",
        "volume": 512,
        "pages": "661-674",
        "year": 2020,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        has_archive=True,
        supports_candidate_injection=True,
        supports_restart=False,
        supports_discrete=True,
        supports_integer=False,
        supports_mixed=False,
        supports_diversity_metrics=True,
        supports_snapshot_fit=True,
    )

    # Paper settings: N=100, uniform crossover probability 0.5, bit-flip
    # mutation probability 1/D, lambda=0.05, and 12 bits per real variable.
    # Here D is the binary chromosome length after real-variable encoding.
    _DEFAULTS: dict[str, Any] = {
        "population_size": 100,
        "uniform_crossover_probability": 0.5,
        "bit_flip_probability": None,
        "learning_probability": 0.05,
        "bits_per_real": 12,
        "encoding": "auto",  # auto | real | binary
    }

    _OPERATOR_LABELS = (
        "bspga.uniform_crossover",
        "bspga.bit_flip_mutation",
        "bspga.tree_learning",
        "bspga.tree_collision_fine_tuning",
        "bspga.tree_insertion",
        "bspga.environmental_selection",
        "bspga.candidate_injection",
    )

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        raw = dict(config.params or {})
        p = {**self._DEFAULTS, **raw}

        # Backward-compatible aliases from the former non-paper implementation.
        if "crossover_rate" in raw and "uniform_crossover_probability" not in raw:
            p["uniform_crossover_probability"] = raw["crossover_rate"]
        if "mutation_rate" in raw and "bit_flip_probability" not in raw:
            p["bit_flip_probability"] = raw["mutation_rate"]
        if "bsp_learning_rate" in raw and "learning_probability" not in raw:
            p["learning_probability"] = raw["bsp_learning_rate"]

        self._requested_n = max(1, int(p["population_size"]))
        self._pc = float(p["uniform_crossover_probability"])
        self._lambda = float(p["learning_probability"])
        self._bits_per_real = int(p["bits_per_real"])
        self._encoding = self._resolve_encoding(str(p.get("encoding", "auto")))

        if not 0.0 <= self._pc <= 1.0:
            raise ValueError("bspga uniform_crossover_probability must be in [0, 1].")
        if not 0.0 <= self._lambda <= 1.0:
            raise ValueError("bspga learning_probability must be in [0, 1].")
        if self._bits_per_real < 1:
            raise ValueError("bspga bits_per_real must be >= 1.")

        self._dimension = int(problem.dimension)
        self._lo = np.asarray(problem.min_values, dtype=float)
        self._hi = np.asarray(problem.max_values, dtype=float)
        if self._lo.shape != self._hi.shape or self._lo.size != self._dimension:
            raise ValueError("bspga bounds must match the problem dimension.")
        if np.any(~np.isfinite(self._lo)) or np.any(~np.isfinite(self._hi)):
            raise ValueError("bspga requires finite bounds.")
        if np.any(self._hi < self._lo):
            raise ValueError("bspga requires max_values >= min_values.")

        self._chromosome_length = (
            self._dimension if self._encoding == "binary"
            else self._dimension * self._bits_per_real
        )
        if self._chromosome_length < 1:
            raise ValueError("bspga requires at least one binary chromosome bit.")

        # A binary search space cannot contain more unique individuals than
        # 2^L.  The cap only matters for tiny explicitly binary problems.
        if self._encoding == "binary" and self._chromosome_length < 63:
            self._n = min(self._requested_n, 1 << self._chromosome_length)
        else:
            self._n = self._requested_n

        configured_pm = p.get("bit_flip_probability")
        self._pm = (
            1.0 / float(self._chromosome_length)
            if configured_pm is None
            else float(configured_pm)
        )
        if not 0.0 <= self._pm <= 1.0:
            raise ValueError("bspga bit_flip_probability must be in [0, 1].")

        legacy_keys = {
            "mutation_scale",
            "elite_fraction",
            "tournament_size",
            "bsp_neighbor_count",
        }
        self._ignored_legacy_params = sorted(legacy_keys.intersection(raw))

        if config.seed is not None:
            np.random.seed(config.seed)

    # ------------------------------------------------------------------
    # Representation and objective evaluation
    # ------------------------------------------------------------------

    def _resolve_encoding(self, requested: str) -> str:
        mode = requested.strip().lower()
        if mode not in {"auto", "real", "binary"}:
            raise ValueError("bspga encoding must be 'auto', 'real', or 'binary'.")
        descriptors = self.problem._normalize_variable_types()
        kinds: list[str] = []
        for descriptor in descriptors:
            if descriptor is None:
                kinds.append("real")
            elif isinstance(descriptor, dict):
                kinds.append(str(descriptor.get("type", descriptor.get("kind", "real"))).lower())
            else:
                kinds.append(str(descriptor).strip().lower())

        binary_tokens = {"binary", "bool", "boolean"}
        real_tokens = {"", "float", "real", "continuous", "double"}
        all_binary = all(kind in binary_tokens for kind in kinds)
        all_real = all(kind in real_tokens for kind in kinds)

        if mode == "auto":
            if all_binary:
                return "binary"
            if all_real:
                return "real"
            raise ValueError(
                "bspga paper supports homogeneous binary or real variables; "
                "mixed/integer variable_types are not a native BSPGA domain."
            )
        if mode == "binary" and not all_binary and self.problem.variable_types is not None:
            raise ValueError("bspga binary encoding conflicts with non-binary variable_types.")
        if mode == "real" and not all_real:
            raise ValueError("bspga real encoding conflicts with discrete variable_types.")
        return mode

    def _encode_position(self, position: np.ndarray | list[float]) -> np.ndarray:
        pos = np.asarray(position, dtype=float)
        if pos.size != self._dimension:
            raise ValueError(f"Expected position dimension {self._dimension}, got {pos.size}.")
        pos = np.clip(pos, self._lo, self._hi)
        if self._encoding == "binary":
            projected = self.problem.apply_variable_types(pos)
            return (np.asarray(projected, dtype=float) >= 0.5).astype(np.uint8)

        levels = (1 << self._bits_per_real) - 1
        span = self._hi - self._lo
        normalized = np.divide(
            pos - self._lo,
            span,
            out=np.zeros_like(pos, dtype=float),
            where=span > 0.0,
        )
        integers = np.rint(np.clip(normalized, 0.0, 1.0) * levels).astype(np.int64)
        shifts = np.arange(self._bits_per_real, dtype=np.int64)
        # The paper's decoding equation uses b_j * 2^(j-1), hence little-endian
        # bit order inside each real variable's 12-bit block.
        return ((integers[:, None] >> shifts[None, :]) & 1).astype(np.uint8).reshape(-1)

    def _decode_bits(self, bits: np.ndarray) -> np.ndarray:
        chromosome = np.asarray(bits, dtype=np.uint8).reshape(-1)
        if chromosome.size != self._chromosome_length:
            raise ValueError(
                f"Expected chromosome length {self._chromosome_length}, got {chromosome.size}."
            )
        if self._encoding == "binary":
            return self.problem.apply_variable_types(chromosome.astype(float))

        blocks = chromosome.reshape(self._dimension, self._bits_per_real).astype(np.int64)
        weights = (1 << np.arange(self._bits_per_real, dtype=np.int64))
        integers = blocks @ weights
        levels = float((1 << self._bits_per_real) - 1)
        position = self._lo + (self._hi - self._lo) * (integers.astype(float) / levels)
        return self.problem.apply_variable_types(position)

    def _evaluate_bits(self, bits: np.ndarray) -> tuple[np.ndarray, float]:
        position = self._decode_bits(bits)
        details = self.problem.evaluate_details(position, apply_handler=True)
        return np.asarray(details["position"], dtype=float), float(details["fitness"])

    def _initial_chromosomes(self) -> np.ndarray:
        # The paper initializes binary variables directly.  The uniform-position
        # branch is used only when the package's custom init_function hook is
        # active, because that hook supplies decoded positions by intercepting
        # np.random.uniform during initialize().
        if callable(getattr(self.config, "init_function", None)):
            positions = np.random.uniform(self._lo, self._hi, (self._n, self._dimension))
            return np.vstack([self._encode_position(row) for row in positions]).astype(np.uint8)
        return np.random.randint(
            0, 2, size=(self._n, self._chromosome_length), dtype=np.uint8
        )

    # ------------------------------------------------------------------
    # BSP tree (Algorithms 2 and 3)
    # ------------------------------------------------------------------

    def _empty_tree(self) -> dict[str, Any]:
        return {
            "root": None,
            "best_bits": None,
            "best_position": None,
            "best_fitness": None,
            "best_level": None,
            "insertions": 0,
            "node_count": 0,
            "collisions": 0,
            "new_solution_flips": 0,
            "stored_solution_flips": 0,
            "stored_solution_reevaluations": 0,
            "exhausted_reuses": 0,
        }

    def _tree_update_best(self, tree: dict[str, Any], node: _BSPNode) -> bool:
        incumbent = tree.get("best_fitness")
        if incumbent is None or self.problem.is_better(float(node.fitness), float(incumbent)):
            tree["best_bits"] = node.bits.copy()
            tree["best_position"] = node.position.copy()
            tree["best_fitness"] = float(node.fitness)
            tree["best_level"] = int(node.level)
            return True
        return False

    def _fitness_gain(self, reference: float, candidate: float) -> float:
        if self.problem.objective == "min":
            return max(0.0, float(reference) - float(candidate))
        return max(0.0, float(candidate) - float(reference))

    def _insert_into_tree(
        self,
        tree: dict[str, Any],
        bits: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float, int, dict[str, Any]]:
        """Insert one candidate exactly in the order of Algorithm 2."""
        candidate_bits = np.asarray(bits, dtype=np.uint8).copy()
        before_best = tree.get("best_fitness")
        event = {
            "collision": False,
            "new_solution_flipped": False,
            "stored_solution_flipped": False,
            "duplicate_reused": False,
            "archive_gain": 0.0,
        }

        root = tree.get("root")
        if root is None:
            position, fitness = self._evaluate_bits(candidate_bits)
            node = _BSPNode(candidate_bits.copy(), position.copy(), fitness, level=1)
            tree["root"] = node
            tree["insertions"] = 1
            tree["node_count"] = 1
            self._tree_update_best(tree, node)
            return candidate_bits, position, fitness, 1, event

        leaf: _BSPNode = root
        while not leaf.is_leaf:
            bit_index = leaf.level - 1
            if bit_index >= self._chromosome_length:
                break
            leaf = leaf.left if candidate_bits[bit_index] == 0 else leaf.right
            if leaf is None:  # defensive; a valid BSPGA tree always has both children
                raise RuntimeError("Corrupted BSPGA tree: missing child in an internal node.")

        bit_index = leaf.level - 1
        if bit_index >= self._chromosome_length:
            # The finite binary space at this path has been exhausted.  Reuse the
            # recorded value without reevaluation, preserving non-revisiting.
            tree["exhausted_reuses"] += 1
            event["duplicate_reused"] = True
            return (
                leaf.bits.copy(),
                leaf.position.copy(),
                float(leaf.fitness),
                0,
                event,
            )

        existing_bits = leaf.bits.copy()
        existing_position = leaf.position.copy()
        existing_fitness = float(leaf.fitness)
        evaluations = 0

        if int(existing_bits[bit_index]) == int(candidate_bits[bit_index]):
            event["collision"] = True
            tree["collisions"] += 1
            best_bits = np.asarray(tree["best_bits"], dtype=np.uint8)
            dis_new = int(np.count_nonzero(candidate_bits != best_bits))
            dis_existing = int(np.count_nonzero(existing_bits != best_bits))

            # Algorithm 2 flips the solution with smaller Hamming distance to
            # the current best.  Ties go to the stored solution (the paper's
            # explicit else branch).
            if dis_new < dis_existing:
                candidate_bits[bit_index] ^= np.uint8(1)
                event["new_solution_flipped"] = True
                tree["new_solution_flips"] += 1
            else:
                existing_bits[bit_index] ^= np.uint8(1)
                existing_position, existing_fitness = self._evaluate_bits(existing_bits)
                evaluations += 1
                event["stored_solution_flipped"] = True
                tree["stored_solution_flips"] += 1
                tree["stored_solution_reevaluations"] += 1

        candidate_position, candidate_fitness = self._evaluate_bits(candidate_bits)
        evaluations += 1

        child_level = leaf.level + 1
        existing_child = _BSPNode(
            existing_bits.copy(),
            existing_position.copy(),
            float(existing_fitness),
            level=child_level,
        )
        candidate_child = _BSPNode(
            candidate_bits.copy(),
            candidate_position.copy(),
            float(candidate_fitness),
            level=child_level,
        )

        if int(existing_bits[bit_index]) == 0:
            leaf.left, leaf.right = existing_child, candidate_child
        else:
            leaf.left, leaf.right = candidate_child, existing_child

        tree["insertions"] += 1
        tree["node_count"] += 2
        self._tree_update_best(tree, existing_child)
        self._tree_update_best(tree, candidate_child)
        after_best = float(tree["best_fitness"])
        if before_best is not None:
            event["archive_gain"] = self._fitness_gain(float(before_best), after_best)

        return candidate_bits, candidate_position, candidate_fitness, evaluations, event

    def _tree_learning(
        self,
        offspring: np.ndarray,
        tree: dict[str, Any],
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        learned = np.asarray(offspring, dtype=np.uint8).copy()
        details: list[dict[str, Any]] = []
        best_bits = tree.get("best_bits")
        best_level = tree.get("best_level")
        prefix = 0 if best_level is None else max(0, min(self._chromosome_length, int(best_level) - 1))

        for i in range(learned.shape[0]):
            applied = bool(np.random.random() < self._lambda)
            changed = 0
            if applied and prefix > 0 and best_bits is not None:
                before = learned[i, :prefix].copy()
                learned[i, :prefix] = np.asarray(best_bits, dtype=np.uint8)[:prefix]
                changed = int(np.count_nonzero(before != learned[i, :prefix]))
            details.append({"applied": applied, "prefix": prefix, "changed_bits": changed})
        return learned, details

    # ------------------------------------------------------------------
    # Genetic operators and environmental selection (Algorithm 1)
    # ------------------------------------------------------------------

    def _genetic_operators(
        self,
        population_bits: np.ndarray,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        n = int(population_bits.shape[0])
        order = np.random.permutation(n)
        children: list[np.ndarray] = []
        lineage: list[dict[str, Any]] = []

        cursor = 0
        while len(children) < n:
            p1_idx = int(order[cursor % n])
            if n == 1:
                p2_idx = p1_idx
            else:
                p2_idx = int(order[(cursor + 1) % n])
            p1 = population_bits[p1_idx]
            p2 = population_bits[p2_idx]
            mask = np.random.random(self._chromosome_length) < self._pc
            pair = (
                np.where(mask, p1, p2).astype(np.uint8),
                np.where(mask, p2, p1).astype(np.uint8),
            )
            for child in pair:
                if len(children) >= n:
                    break
                mutation_mask = np.random.random(self._chromosome_length) < self._pm
                child = child.copy()
                child[mutation_mask] ^= np.uint8(1)
                children.append(child)
                lineage.append(
                    {
                        "parent_indices": [p1_idx, p2_idx],
                        "crossover_parent1_bits": int(np.count_nonzero(mask)),
                        "crossover_parent2_bits": int(self._chromosome_length - np.count_nonzero(mask)),
                        "mutated_bits": int(np.count_nonzero(mutation_mask)),
                    }
                )
            cursor += 2

        return np.vstack(children).astype(np.uint8), lineage

    def _order(self, fitness: np.ndarray) -> np.ndarray:
        indices = np.argsort(np.asarray(fitness, dtype=float), kind="stable")
        return indices if self.problem.objective == "min" else indices[::-1]

    def initialize(self) -> EngineState:
        chromosomes = self._initial_chromosomes()
        tree = self._empty_tree()
        pop_bits: list[np.ndarray] = []
        pop_rows: list[np.ndarray] = []
        evaluations = 0

        for chromosome in chromosomes:
            final_bits, position, fitness, used, _ = self._insert_into_tree(tree, chromosome)
            evaluations += used
            pop_bits.append(final_bits)
            pop_rows.append(np.concatenate((position, [fitness])))

        population = np.vstack(pop_rows).astype(float)
        population_bits = np.vstack(pop_bits).astype(np.uint8)
        best_position = np.asarray(tree["best_position"], dtype=float)
        best_fitness = float(tree["best_fitness"])

        return EngineState(
            step=0,
            evaluations=evaluations,
            best_position=best_position.tolist(),
            best_fitness=best_fitness,
            initialized=True,
            payload={
                "population": population,
                "population_bits": population_bits,
                "tree": tree,
                "lineage": [],
                "operator_counts": {label: 0 for label in self._OPERATOR_LABELS},
                "operator_contributions": {label: 0.0 for label in self._OPERATOR_LABELS},
                "selected_offspring": 0,
                "mutation_bits": 0,
                "learning_events": 0,
                "learning_changed_bits": 0,
                "generation_tree_evaluations": 0,
            },
        )

    def step(self, state: EngineState) -> EngineState:
        population = np.asarray(state.payload["population"], dtype=float).copy()
        population_bits = np.asarray(state.payload["population_bits"], dtype=np.uint8).copy()
        tree = state.payload["tree"]
        n = int(population.shape[0])

        offspring_bits, lineage = self._genetic_operators(population_bits)
        offspring_bits, learning = self._tree_learning(offspring_bits, tree)

        offspring_rows: list[np.ndarray] = []
        final_offspring_bits: list[np.ndarray] = []
        child_info: list[dict[str, Any]] = []
        evaluations = 0
        counts = {label: 0 for label in self._OPERATOR_LABELS}
        contributions = {label: 0.0 for label in self._OPERATOR_LABELS}

        counts["bspga.uniform_crossover"] = n
        counts["bspga.bit_flip_mutation"] = n
        mutation_bits = int(sum(item["mutated_bits"] for item in lineage))
        learning_events = int(sum(bool(item["applied"]) for item in learning))
        learning_changed_bits = int(sum(item["changed_bits"] for item in learning))
        counts["bspga.tree_learning"] = learning_events

        for i in range(n):
            parent_indices = lineage[i]["parent_indices"]
            parent_fitnesses = population[parent_indices, -1]
            parent_best = float(parent_fitnesses[self._order(parent_fitnesses)[0]])

            final_bits, position, fitness, used, event = self._insert_into_tree(
                tree, offspring_bits[i]
            )
            evaluations += used
            counts["bspga.tree_insertion"] += 1
            archive_gain = float(event["archive_gain"])
            if event["collision"]:
                counts["bspga.tree_collision_fine_tuning"] += 1
                contributions["bspga.tree_collision_fine_tuning"] += 0.5 * archive_gain
                contributions["bspga.tree_insertion"] += 0.5 * archive_gain
            else:
                contributions["bspga.tree_insertion"] += archive_gain

            active = ["bspga.uniform_crossover"]
            if lineage[i]["mutated_bits"] > 0:
                active.append("bspga.bit_flip_mutation")
            if learning[i]["changed_bits"] > 0:
                active.append("bspga.tree_learning")
            if event["new_solution_flipped"]:
                active.append("bspga.tree_collision_fine_tuning")

            final_offspring_bits.append(final_bits)
            offspring_rows.append(np.concatenate((position, [fitness])))
            child_info.append(
                {
                    **lineage[i],
                    **learning[i],
                    **event,
                    "parent_best_fitness": parent_best,
                    "child_fitness": float(fitness),
                    "active_operators": active,
                    "tree_evaluations": int(used),
                }
            )

        offspring = np.vstack(offspring_rows).astype(float)
        final_bits_matrix = np.vstack(final_offspring_bits).astype(np.uint8)

        combined = np.vstack((population, offspring))
        combined_bits = np.vstack((population_bits, final_bits_matrix))
        selected_indices = self._order(combined[:, -1])[:n]
        next_population = combined[selected_indices].copy()
        next_bits = combined_bits[selected_indices].copy()

        selected_offspring = 0
        for idx in selected_indices:
            if int(idx) < n:
                continue
            selected_offspring += 1
            info = child_info[int(idx) - n]
            gain = self._fitness_gain(info["parent_best_fitness"], info["child_fitness"])
            active = list(info["active_operators"])
            if gain > 0.0:
                credited = active + ["bspga.environmental_selection"]
                share = gain / float(len(credited))
                for label in credited:
                    contributions[label] += share
        counts["bspga.environmental_selection"] = 1

        for i, info in enumerate(child_info):
            info["id"] = f"bspga:{state.step + 1}:{i}"
            info["selected"] = bool((n + i) in set(int(v) for v in selected_indices))

        state.step += 1
        state.evaluations += evaluations
        state.best_position = np.asarray(tree["best_position"], dtype=float).tolist()
        state.best_fitness = float(tree["best_fitness"])
        state.payload = {
            "population": next_population,
            "population_bits": next_bits,
            "tree": tree,
            "lineage": child_info,
            "operator_counts": counts,
            "operator_contributions": contributions,
            "selected_offspring": int(selected_offspring),
            "mutation_bits": mutation_bits,
            "learning_events": learning_events,
            "learning_changed_bits": learning_changed_bits,
            "generation_tree_evaluations": int(evaluations),
        }
        return state

    # ------------------------------------------------------------------
    # Package interoperability and telemetry
    # ------------------------------------------------------------------

    def observe(self, state: EngineState) -> dict[str, Any]:
        population = np.asarray(state.payload["population"], dtype=float)
        positions = population[:, :-1]
        fitness = population[:, -1]
        span_norm = float(np.linalg.norm(self._hi - self._lo)) or 1.0
        centroid = np.mean(positions, axis=0)
        diversity = float(np.mean(np.linalg.norm(positions - centroid, axis=1)) / span_norm)
        tree = state.payload["tree"]

        return {
            "step": int(state.step),
            "evaluations": int(state.evaluations),
            "best_fitness": float(state.best_fitness),
            "mean_fitness": float(np.mean(fitness)),
            "std_fitness": float(np.std(fitness)),
            "diversity": diversity,
            "population_size": int(population.shape[0]),
            "encoding": self._encoding,
            "chromosome_length": int(self._chromosome_length),
            "bits_per_real": int(self._bits_per_real if self._encoding == "real" else 1),
            "learning_probability": float(self._lambda),
            "mutation_probability": float(self._pm),
            "tree_insertions": int(tree["insertions"]),
            "tree_node_count": int(tree["node_count"]),
            "tree_collisions": int(tree["collisions"]),
            "stored_solution_reevaluations": int(tree["stored_solution_reevaluations"]),
            "selected_offspring": int(state.payload.get("selected_offspring", 0)),
            "mutation_bits": int(state.payload.get("mutation_bits", 0)),
            "learning_events": int(state.payload.get("learning_events", 0)),
            "learning_changed_bits": int(state.payload.get("learning_changed_bits", 0)),
            "generation_tree_evaluations": int(state.payload.get("generation_tree_evaluations", 0)),
            "operator_counts": dict(state.payload.get("operator_counts", {})),
            "operator_contributions": dict(state.payload.get("operator_contributions", {})),
            "evomapx_operator_labels": list(self._OPERATOR_LABELS[:-1]),
            "native_evomapx_operator_labels": True,
            "evomapx_delta_f": "improvement_positive",
            "evomapx_fidelity": "native",
        }

    def get_best_candidate(self, state: EngineState) -> CandidateRecord:
        return CandidateRecord(
            position=list(state.best_position),
            fitness=float(state.best_fitness),
            source_algorithm=self.algorithm_id,
            source_step=state.step,
            role="best",
            metadata={"source": "bsp_tree_archive"},
        )

    def get_population(self, state: EngineState) -> list[CandidateRecord]:
        population = np.asarray(state.payload["population"], dtype=float)
        return [
            CandidateRecord(
                position=population[i, :-1].tolist(),
                fitness=float(population[i, -1]),
                source_algorithm=self.algorithm_id,
                source_step=state.step,
                role="current",
            )
            for i in range(population.shape[0])
        ]

    def inject_candidates(
        self,
        state: EngineState,
        candidates: list[CandidateRecord],
        policy: str = "native",
    ) -> EngineState:
        """Package extension: archive incoming candidates, then elitistically merge."""
        if not candidates:
            return state
        population = np.asarray(state.payload["population"], dtype=float).copy()
        population_bits = np.asarray(state.payload["population_bits"], dtype=np.uint8).copy()
        tree = state.payload["tree"]
        evaluations = 0
        accepted = 0
        total_gain = 0.0

        for candidate in candidates:
            bits = self._encode_position(candidate.position)
            final_bits, position, fitness, used, event = self._insert_into_tree(tree, bits)
            evaluations += used
            worst_index = int(self._order(population[:, -1])[-1])
            worst_fitness = float(population[worst_index, -1])
            if self.problem.is_better(float(fitness), worst_fitness) or float(fitness) == worst_fitness:
                total_gain += self._fitness_gain(worst_fitness, float(fitness))
                population[worst_index, :-1] = position
                population[worst_index, -1] = fitness
                population_bits[worst_index] = final_bits
                accepted += 1
            total_gain += float(event["archive_gain"])

        state.evaluations += evaluations
        state.best_position = np.asarray(tree["best_position"], dtype=float).tolist()
        state.best_fitness = float(tree["best_fitness"])
        state.payload["population"] = population
        state.payload["population_bits"] = population_bits
        state.payload["tree"] = tree
        state.payload["operator_counts"] = {
            label: (len(candidates) if label == "bspga.candidate_injection" else 0)
            for label in self._OPERATOR_LABELS
        }
        state.payload["operator_counts"]["bspga.tree_insertion"] = len(candidates)
        state.payload["operator_contributions"] = {
            label: (float(total_gain) if label == "bspga.candidate_injection" else 0.0)
            for label in self._OPERATOR_LABELS
        }
        state.payload["selected_offspring"] = int(accepted)
        state.payload["generation_tree_evaluations"] = int(evaluations)
        return state

    def finalize(self, state: EngineState) -> OptimizationResult:
        tree = state.payload["tree"]
        return OptimizationResult(
            algorithm_id=self.algorithm_id,
            best_position=list(state.best_position),
            best_fitness=float(state.best_fitness),
            steps=int(state.step),
            evaluations=int(state.evaluations),
            termination_reason=state.termination_reason,
            capabilities=self.capabilities,
            metadata={
                "algorithm_name": self.algorithm_name,
                "family": self.family,
                "reference": dict(self._REFERENCE),
                "elapsed_time": float(state.elapsed_time),
                "requested_population_size": int(self._requested_n),
                "population_size": int(self._n),
                "encoding": self._encoding,
                "bits_per_real": int(self._bits_per_real if self._encoding == "real" else 1),
                "chromosome_length": int(self._chromosome_length),
                "uniform_crossover_probability": float(self._pc),
                "bit_flip_probability": float(self._pm),
                "learning_probability": float(self._lambda),
                "tree_insertions": int(tree["insertions"]),
                "tree_node_count": int(tree["node_count"]),
                "tree_collisions": int(tree["collisions"]),
                "new_solution_flips": int(tree["new_solution_flips"]),
                "stored_solution_flips": int(tree["stored_solution_flips"]),
                "stored_solution_reevaluations": int(tree["stored_solution_reevaluations"]),
                "exhausted_reuses": int(tree["exhausted_reuses"]),
                "best_source": "bsp_tree_archive",
                "paper_faithful_core": True,
                "ignored_legacy_params": list(self._ignored_legacy_params),
                "evomapx_operator_labels": list(self._OPERATOR_LABELS),
            },
        )

"""Scientific benchmark-study utilities."""

from .problems import BenchmarkProblem, Problem, ProblemSuite
from .records import BenchmarkResult, ExperimentRecord, load_benchmark
from .statistics import cliffs_delta, friedman_test, rank_table, summary_table, wilcoxon_pairwise
from .study import BenchmarkStudy
from .plots import plot_convergence, plot_ecdf, plot_performance_profile, plot_rank_heatmap

__all__ = [
    "BenchmarkProblem",
    "Problem",
    "ProblemSuite",
    "BenchmarkResult",
    "ExperimentRecord",
    "BenchmarkStudy",
    "load_benchmark",
    "summary_table",
    "rank_table",
    "friedman_test",
    "wilcoxon_pairwise",
    "cliffs_delta",
    "plot_convergence",
    "plot_ecdf",
    "plot_performance_profile",
    "plot_rank_heatmap",
]

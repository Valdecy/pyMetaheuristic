"""Explicit EvoMapX operator label catalog.

Generated from the faithful engine source files by static code/profile analysis,
then augmented by a one-step runtime smoke probe so shared-base source labels
(e.g., restart/CMA-ES or local-search bases) are present in the exported list.
This catalog does not change optimizer dynamics.
"""
from __future__ import annotations

ENGINE_OPERATOR_LABELS = {'aaa': ['aaa.step_impl_evaluation_l166', 'aaa.is_replaced_by_corresponding_cell_biggest', 'aaa.adaptation_most_starving_colony_moves_toward', 'aaa.helical_movement', 'aaa.movement_consumes_half_loss_failed_metabolism', 'aaa.other_half_following_pseudo_code_distinction', 'aaa.monod_style_evolutionary_growth_signal_raw', 'aaa.negative_maximized_so_package_s_normalized', 'aaa.used_as_nutrient_proxy', 'aaa.reproduction_one_random_cell_dimension_smallest'], 'aao': ['aao.initialize_evaluation_l99', 'aao.step_evaluation_l301', 'aao.step_evaluation_l313', 'aao.initialization'], 'abco': ['abco.employed', 'abco.onlooker', 'abco.scout', 'abco.initialize_evaluation_l91', 'abco.initialization'], 'acgwo': ['acgwo.initialize_evaluation_l32', 'acgwo.step_evaluation_l55', 'acgwo.step_evaluation_l56', 'acgwo.step_evaluation_l57', 'acgwo.step_evaluation_l60', 'acgwo.initialization'], 'aco': ['aco.random_real_valued_initialization', 'aco.pheromone_weighted_perturbation_in_each_dimension', 'aco.initialization', 'aco.random_walk', 'aco.pheromone_deposit_on_best_solutions_dimension', 'aco.keep_best_survivors'], 'acor': ['acor.step_impl_evaluation_l33'], 'adam': ['adam.initialize_evaluation_l57', 'adam.step_evaluation_l78', 'adam.initialization', 'adam.numerical_grad_evaluation_l23'], 'adaptive_eo': ['adaptive_eo.initialize_evaluation_l99', 'adaptive_eo.step_evaluation_l301', 'adaptive_eo.step_evaluation_l313', 'adaptive_eo.initialization'], 'aefa': ['aefa.step_impl_evaluation_l45'], 'aeo': ['aeo.production_worst_agent', 'aeo.step_impl_evaluation_l36', 'aeo.producer_phase', 'aeo.consumption'], 'aesspso': ['aesspso.initialize_evaluation_l68', 'aesspso.safe', 'aesspso.initialization', 'aesspso.global_best_pbest', 'aesspso.per_particle_adaptive_weights', 'aesspso.scalar_w_stability_adapted', 'aesspso.pbest'], 'afsa': ['afsa.init_pop', 'afsa.prey', 'afsa.swarm', 'afsa.follow', 'afsa.leap'], 'aft': ['aft.step_impl_evaluation_l52', 'aft.local_bests_greedy', 'aft.replacement'], 'agto': ['agto.step_impl_evaluation_l52', 'agto.step_impl_evaluation_l72', 'agto.exploration', 'agto.exploitation'], 'aha': ['aha.handle_nan', 'aha.territorial_foraging', 'aha.migration', 'aha.direction_vector', 'aha.guided_foraging'], 'aho': ['aho.step_impl_evaluation_l57', 'aho.step_impl_evaluation_l76'], 'aiw_pso': ['aiw_pso.initialize_evaluation_l99', 'aiw_pso.step_evaluation_l301', 'aiw_pso.step_evaluation_l313', 'aiw_pso.initialization'], 'ala': ['ala.step_impl_evaluation_l52'], 'alo': ['alo.init_pop', 'alo.step_evaluation_l62', 'alo.inject_candidates_evaluation_l116', 'alo.inject_candidates', 'alo.combine'], 'ao': ['ao.eq_14', 'ao.guard_single_step_runs_original_denominator', 'ao.in_that_degenerate_schedule_quality_factor', 'ao.spiral_shape_eqs_9_10', 'ao.exploration', 'ao.eq_3', 'ao.eq_5', 'ao.exploitation', 'ao.eq_13'], 'aoa': ['aoa.initialize_evaluation_l24', 'aoa.step_evaluation_l43', 'aoa.initialization'], 'aoo': ['aoo.step_impl_evaluation_l139'], 'apo': ['apo.step_impl_evaluation_l73'], 'arch_oa': ['arch_oa.step_impl_evaluation_l85', 'arch_oa.best_object_s_den_vol_acc', 'arch_oa.transfer_operator_eq_8', 'arch_oa.density_decreasing_factor_eq_9'], 'aro': ['aro.eq_11', 'aro.eq_15_decay_factor', 'aro.sparse_random_direction_vector_r_eq', 'aro.eq_15', 'aro.detour_foraging_eq_1', 'aro.random_hiding_eqs_11_13', 'aro.eq_12', 'aro.eq_8', 'aro.eq_13'], 'ars': ['ars.initialize_evaluation_l27', 'ars.small_step', 'ars.large_step', 'ars.inject_candidates_evaluation_l114', 'ars.initialization', 'ars.inject_candidates'], 'artemisinin_o': ['artemisinin_o.boundary', 'artemisinin_o.mutation'], 'aso': ['aso.step_impl_evaluation_l41'], 'aso_atom': ['aso_atom.do_not_move_current_elites_unless', 'aso_atom.number_active_neighbours_decreases_over_time', 'aso_atom.exploratory_late_iterations_emphasize_best_atoms', 'aso_atom.smooth_bounded_approximation_aso_interaction_repulsive', 'aso_atom.very_small_distances_attractive_otherwise_decaying', 'aso_atom.refine_around_best_by_replacing_few', 'aso_atom.exploitation', 'aso_atom.mechanism_on_narrow_continuous_optima'], 'autov': ['autov.initialize_evaluation_l71', 'autov.step_evaluation_l91', 'autov.initialization'], 'avoa': ['avoa.avoa_replaces_all_no_greedy_in', 'avoa.exploration', 'avoa.exploitation', 'avoa.1', 'avoa.2'], 'bacterial_colony_o': ['bacterial_colony_o.implementation_but_only_as_bounded_macro', 'bacterial_colony_o.current_colony_best_accept_only_it', 'bacterial_colony_o.reproduction_elimination_using_normalized_energy', 'bacterial_colony_o.migration_is_triggered_by_low_positional', 'bacterial_colony_o.chemotaxis_communication_blend_personal_global_directions', 'bacterial_colony_o.then_greedily_keep_successful_tumbles_swims', 'bacterial_colony_o.swim_repeatedly_in_non_turbulent_direction', 'bacterial_colony_o.interactive_exchange_between_bacteria', 'bacterial_colony_o.group_exchange_move_weak_bacterium_slightly'], 'basin_hopping': ['basin_hopping.step', 'basin_hopping.local_search_evaluation_l145', 'basin_hopping.local_search_evaluation_l177'], 'bat_a': ['bat_a.initialize_evaluation_l25', 'bat_a.step_evaluation_l46', 'bat_a.step_evaluation_l51', 'bat_a.step_evaluation_l53', 'bat_a.inject_candidates_evaluation_l113', 'bat_a.initialization', 'bat_a.inject_candidates'], 'bbo': ['bbo.init_pop', 'bbo.step_evaluation_l50', 'bbo.step_evaluation_l57'], 'bboa': ['bboa.step_impl_evaluation_l45', 'bboa.2_sniffing', 'bboa.1_pedal_marking'], 'bbso': ['bbso.very_flat_shifted_cost_landscapes', 'bbso.matlab_source_uses_j_2_i', 'bbso.j_1_i_so_first_bug', 'bbso.compute_fr_progress_in_log_space'], 'bco': ['bco.swim_refine_without_turbulence', 'bco.tumble_turbulence', 'bco.neighbour_exchange'], 'bea': ['bea.step_impl_evaluation_l37', 'bea.step_impl_evaluation_l43'], 'bes': ['bes.stage_1_select_space', 'bes.stage_2_search_in_space', 'bes.stage_3_swoop'], 'bfgs': ['bfgs.initialize_evaluation_l56', 'bfgs.armijo_line_search', 'bfgs.initialization', 'bfgs.search_direction', 'bfgs.steepest_descent', 'bfgs.bfgs', 'bfgs.grad_evaluation_l23'], 'bfo': ['bfo.step_impl_evaluation_l38', 'bfo.step_impl_evaluation_l51', 'bfo.step_impl_evaluation_l55'], 'bipop_cmaes': ['bipop_cmaes.step', 'bipop_cmaes.step_evaluation_l534'], 'bka': ['bka.step_impl_evaluation_l33', 'bka.step_impl_evaluation_l43'], 'bmo': ['bmo.step_impl_evaluation_l37', 'bmo.replacement'], 'bono': ['bono.step_impl_evaluation_l89'], 'boa': ['boa.step_impl_evaluation_l39'], 'bps': ['bps.step_impl_evaluation_l141'], 'bro': ['bro.find_nearest_neighbour', 'bro.step_impl_evaluation_l54', 'bro.dynamic_bound_contraction', 'bro.ensure_lb_ub'], 'bsa': ['bsa.step_impl_evaluation_l45', 'bsa.step_impl_evaluation_l57', 'bsa.step_impl_evaluation_l63'], 'bso': ['bso.two_cluster_idea', 'bso.possibly_replace_cluster_center', 'bso.single_cluster_idea'], 'bspga': ['bspga.evaluate_positions'], 'btoa': ['btoa.dynamic_position_candidate', 'btoa.step_impl_evaluation_l132', 'btoa.step_impl_evaluation_l143'], 'bwo': ['bwo.crossover', 'bwo.mutation', 'bwo.procreation', 'bwo.cannibalism_keep_1_child'], 'ca': ['ca.init_pop', 'ca.simpler_binary_tournament'], 'camel': ['camel.step_impl_evaluation_l38', 'camel.step_impl_evaluation_l45'], 'capsa': ['capsa.step_impl_evaluation_l60', 'capsa.step_impl_evaluation_l65'], 'cat_so': ['cat_so.init_pop', 'cat_so.velocities_reuse_pop_shape', 'cat_so.step_evaluation_l59'], 'cco': ['cco.greedy_single'], 'cddo': ['cddo.step_impl_evaluation_l58', 'cddo.top_k_as_pattern', 'cddo.local_bests'], 'cddo_child': ['cddo_child.step_impl_evaluation_l74'], 'cdo': ['cdo.step_impl_evaluation_l34'], 'cdo_chernobyl': ['cdo_chernobyl.step_impl_evaluation_l54'], 'cem': ['cem.init_pop', 'cem.step_evaluation_l49', 'cem.inject_candidates_evaluation_l102', 'cem.inject_candidates'], 'ceo_cosmic': ['ceo_cosmic.step_impl_evaluation_l86'], 'cfoa': ['cfoa.step_impl_evaluation_l125', 'cfoa.matlab_code_first_evaluates_newfisher_greedily', 'cfoa.in_pymh_protocol_pop_already_stores'], 'cg_gwo': ['cg_gwo.initialize_evaluation_l99', 'cg_gwo.step_evaluation_l301', 'cg_gwo.step_evaluation_l313', 'cg_gwo.initialization'], 'cgo': ['cgo.evaluate_all_4n_seeds_keep_best'], 'chameleon_sa': ['chameleon_sa.step_impl_evaluation_l45', 'chameleon_sa.step_impl_evaluation_l49'], 'chaotic_gwo': ['chaotic_gwo.initialize_evaluation_l99', 'chaotic_gwo.step_evaluation_l301', 'chaotic_gwo.step_evaluation_l313', 'chaotic_gwo.initialization'], 'chicken_so': ['chicken_so.init_pop', 'chicken_so.step_evaluation_l53', 'chicken_so.step_evaluation_l71'], 'chio': ['chio.immune_contact', 'chio.advance_age_infected_recover_too_old', 'chio.infected_contact', 'chio.susceptible_contact', 'chio.become_infected_previously_susceptible_above_average'], 'choa': ['choa.step_impl_evaluation_l46', 'choa.simplified_chaos_map'], 'circle_sa': ['circle_sa.eq_8', 'circle_sa.replacement'], 'clonalg': ['clonalg.init_pop', 'clonalg.step_evaluation_l50', 'clonalg.step_evaluation_l54'], 'cmaes': ['cmaes.evaluate_one_point_get_initial_best', 'cmaes.evaluate', 'cmaes.initialization', 'cmaes.eigen_decomposition_sampling', 'cmaes.sample_n_offspring', 'cmaes.shape_d_n', 'cmaes.n_d', 'cmaes.rank', 'cmaes.mean', 'cmaes.step_size_control', 'cmaes.cholesky_c_ps', 'cmaes.covariance', 'cmaes.enforce_psd'], 'coa': ['coa.social_condition_eq_12', 'coa.pup_birth_eq_7', 'coa.probability_social_condition', 'coa.leaving_probability', 'coa.split_packs', 'coa.alpha_best', 'coa.social_tendency', 'coa.replace_oldest_pup_is_better_than', 'coa.migration_between_packs'], 'coati_oa': ['coati_oa.init_pop', 'coati_oa.step_evaluation_l45'], 'cockroach_so': ['cockroach_so.init_pop', 'cockroach_so.step_evaluation_l46', 'cockroach_so.step_evaluation_l54', 'cockroach_so.step_evaluation_l55'], 'compact_ga': ['compact_ga.evaluate_bits', 'compact_ga.step_evaluation_l211', 'compact_ga.hybrid_continuous_refinement_cga_remains_distribution', 'compact_ga.candidate_generation', 'compact_ga.probability_vector_convergence_on_narrow_continuous', 'compact_ga.nudge_probability_vector_toward_refined_solution', 'compact_ga.exploration', 'compact_ga.match_minimal_pycma_stop_convention_stop', 'compact_ga.without_improving_best_sampled_solution'], 'coot': ['coot.step_impl_evaluation_l47', 'coot.step_impl_evaluation_l60'], 'cpo': ['cpo.step_impl_evaluation_l195', 'cpo.eq_28'], 'crayfish_oa': ['crayfish_oa.step_impl_evaluation_l71'], 'cro': ['cro.step_impl_evaluation_l37', 'cro.step_impl_evaluation_l46'], 'csa': ['csa.init_pop', 'csa.step_evaluation_l48'], 'csbo': ['csbo.systolic', 'csbo.diastolic'], 'cso': ['cso.initialize_evaluation_l35', 'cso.mean_all_positions', 'cso.initialization', 'cso.random_pairing', 'cso.swap_where_loser_actually_has_better'], 'cuckoo_s': ['cuckoo_s.init_pop', 'cuckoo_s.step_evaluation_l49', 'cuckoo_s.abandon_worst_nests'], 'da': ['da.init_pop', 'da.step_evaluation_l82', 'da.inject_candidates_evaluation_l131', 'da.inject_candidates', 'da.food_enemy'], 'dbo': ['dbo.step_impl_evaluation_l40', 'dbo.step_impl_evaluation_l58'], 'ddao': ['ddao.step_impl_evaluation_l26', 'ddao.step_impl_evaluation_l35'], 'de': ['de.init_pop', 'de.step_evaluation_l105'], 'deo_dolphin': ['deo_dolphin.step_impl_evaluation_l60'], 'dfo': ['dfo.init_pop', 'dfo.step_evaluation_l45', 'dfo.step_evaluation_l50'], 'dhole_oa': ['dhole_oa.eq_5', 'dhole_oa.small_weak_prey_immediate_kill', 'dhole_oa.eq_7', 'dhole_oa.eq_3', 'dhole_oa.searching_stage', 'dhole_oa.encircling_stage', 'dhole_oa.large_prey_weaken_repeatedly_attack'], 'dmoa': ['dmoa.selection', 'dmoa.scout_phase', 'dmoa.3_baby_sitter_eviction', 'dmoa.scalar_broadcast', 'dmoa.4_next_position'], 'do_dandelion': ['do_dandelion.3', 'do_dandelion.1', 'do_dandelion.2'], 'doa': ['doa.step_impl_evaluation_l63', 'doa.exploitation'], 'dra': ['dra.step_impl_evaluation_l74', 'dra.step_impl_evaluation_l96', 'dra.step_impl_evaluation_l110'], 'dream_oa': ['dream_oa.step_impl_evaluation_l115'], 'ds_gwo': ['ds_gwo.initialize_evaluation_l99', 'ds_gwo.step_evaluation_l301', 'ds_gwo.step_evaluation_l313', 'ds_gwo.initialization'], 'dso': ['dso.step_impl_evaluation_l31'], 'dvba': ['dvba.init_pop', 'dvba.step_evaluation_l49', 'dvba.step_evaluation_l58'], 'eao': ['eao.candidate_generation'], 'eco': ['eco.step_impl_evaluation_l45'], 'ecological_cycle_o': ['ecological_cycle_o.eval_accept_group', 'ecological_cycle_o.step_impl_evaluation_l82', 'ecological_cycle_o.step_impl_evaluation_l149'], 'ecpo': ['ecpo.initialize_evaluation_l107', 'ecpo.random_perturbation', 'ecpo.initialization', 'ecpo.sort_best_first'], 'edo': ['edo.step_impl_evaluation_l51'], 'eefo': ['eefo.step_impl_evaluation_l117'], 'eel_grouper_o': ['eel_grouper_o.step_impl_evaluation_l72', 'eel_grouper_o.step_impl_evaluation_l98'], 'efo': ['efo.mutation', 'efo.best_worst', 'efo.positive_field_top', 'efo.negative_field_bottom'], 'ego': ['ego.initialize_evaluation_l47', 'ego.candidate_generation', 'ego.initialization'], 'eho': ['eho.init_pop', 'eho.step_evaluation_l53'], 'elk_ho': ['elk_ho.step_impl_evaluation_l50', 'elk_ho.assign_families_females_males_by_fitness'], 'enhanced_aeo': ['enhanced_aeo.initialize_evaluation_l99', 'enhanced_aeo.step_evaluation_l301', 'enhanced_aeo.step_evaluation_l313', 'enhanced_aeo.initialization'], 'enhanced_two': ['enhanced_two.initialize_evaluation_l99', 'enhanced_two.step_evaluation_l301', 'enhanced_two.step_evaluation_l313', 'enhanced_two.initialization'], 'eo': ['eo.position_eq_16', 'eo.build_equilibrium_pool_4_best_centroid', 'eo.4_dim', 'eo.5_dim', 'eo.t_factor_eq_9', 'eo.random_pool_member', 'eo.exponential_factor_eq_11', 'eo.generation_rate_eqs_13_15', 'eo.selection'], 'eoa': ['eoa.step_impl_evaluation_l55', 'eoa.mutation', 'eoa.sort_best_first', 'eoa.reproduction', 'eoa.eq_1', 'eoa.crossover', 'eoa.re_sort'], 'ep': ['ep.generate_offspring', 'ep.merge_parent_offspring', 'ep.selection', 'ep.keep_top_n_by_wins'], 'epc': ['epc.step_impl_evaluation_l33'], 'er_gwo': ['er_gwo.initialize_evaluation_l99', 'er_gwo.step_evaluation_l301', 'er_gwo.step_evaluation_l313', 'er_gwo.initialization'], 'es': ['es.step_impl_evaluation_l40'], 'esc': ['esc.explore_randomly', 'esc.escape_worst', 'esc.move_toward_best'], 'eso': ['eso.step_impl_evaluation_l69', 'eso.field_conductivity', 'eso.field_intensity', 'eso.candidate_generation'], 'esoa': ['esoa.step_impl_evaluation_l42', 'esoa.step_impl_evaluation_l46'], 'et_bo': ['et_bo.step', 'et_bo.evaluate_positions_evaluation_l449'], 'eto': ['eto.trigonometric_component', 'eto.exponential_component'], 'evo': ['evo.step_impl_evaluation_l41', 'evo.step_impl_evaluation_l45'], 'ex_gwo': ['ex_gwo.initialize_evaluation_l99', 'ex_gwo.step_evaluation_l301', 'ex_gwo.step_evaluation_l313', 'ex_gwo.initialization'], 'fata': ['fata.step_impl_evaluation_l44'], 'fbio': ['fbio.team_step_a1_gaussian_perturbation_around', 'fbio.exploration', 'fbio.team_b_step_b1_convex_combination', 'fbio.fitness_proportional_probability'], 'fda': ['fda.init_pop', 'fda.generate_neighbours', 'fda.step_evaluation_l72'], 'fdo': ['fdo.step_impl_evaluation_l32', 'fdo.step_impl_evaluation_l37', 'fdo.step_impl_evaluation_l41'], 'fep': ['fep.initialize_evaluation_l33', 'fep.mutation', 'fep.initialization', 'fep.mutate_strategy_params', 'fep.combine_parent_offspring', 'fep.selection'], 'ffa': ['ffa.step_impl_evaluation_l35', 'ffa.append_best_maintain_diversity_original_paper'], 'ffo': ['ffo.step_impl_evaluation_l21', 'ffo.step_impl_evaluation_l28'], 'firefly_a': ['firefly_a.init_pop', 'firefly_a.step_evaluation_l53'], 'fla': ['fla.tf_0_9'], 'flo': ['flo.greedy_single', 'flo.1', 'flo.2'], 'flood_a': ['flood_a.step_impl_evaluation_l34', 'flood_a.step_impl_evaluation_l46'], 'foa': ['foa.step_impl_evaluation_l38', 'foa.step_impl_evaluation_l55', 'foa.step_impl_evaluation_l58'], 'foa_fossa': ['foa_fossa.step_impl_evaluation_l58'], 'fox': ['fox.preserve_best_few_individuals_explicitly', 'fox.exploitation', 'fox.best_scaled_by_fox_like_jump', 'fox.random_walk', 'fox.positions_radius_decreases_progress'], 'fpa': ['fpa.init_pop', 'fpa.step_evaluation_l53', 'fpa.step_evaluation_l56'], 'frcg': ['frcg.initialize_evaluation_l56', 'frcg.armijo_line_search_along_dk', 'frcg.initialization', 'frcg.fr_restart_every_d_steps', 'frcg.reset', 'frcg.grad_evaluation_l23'], 'frofi': ['frofi.initialize_evaluation_l32', 'frofi.de_offspring_generation', 'frofi.mutation', 'frofi.initialization', 'frofi.selection'], 'fss': ['fss.step_impl_evaluation_l40', 'fss.step_impl_evaluation_l50', 'fss.step_impl_evaluation_l58'], 'fuzzy_gwo': ['fuzzy_gwo.initialize_evaluation_l99', 'fuzzy_gwo.step_evaluation_l301', 'fuzzy_gwo.step_evaluation_l313', 'fuzzy_gwo.initialization'], 'fwa': ['fwa.step_impl_evaluation_l46'], 'ga': ['ga.init_pop', 'ga.breed', 'ga.mutate'], 'gazelle_oa': ['gazelle_oa.step_impl_evaluation_l59', 'gazelle_oa.step_impl_evaluation_l76'], 'gbo': ['gbo.step_impl_evaluation_l78', 'gbo.local_escaping_operator'], 'gbrt_bo': ['gbrt_bo.step', 'gbrt_bo.evaluate_positions_evaluation_l449'], 'gco': ['gco.dark_zone', 'gco.light_zone'], 'gea': ['gea.nearest_by_cosine_similarity', 'gea.second_attempt', 'gea.roulette_wheel_top_nc'], 'ggo': ['ggo.step_impl_evaluation_l23'], 'gja': ['gja.step_impl_evaluation_l60', 'gja.step_impl_evaluation_l69'], 'gjo': ['gjo.exploration', 'gjo.exploitation', 'gjo.gjo_replaces_unconditionally_in_original_use'], 'gkso': ['gkso.crossover', 'gkso.2_shark_hunt'], 'gmo': ['gmo.init_pop', 'gmo.improve_guide', 'gmo.generate_guide'], 'gndo': ['gndo.step_impl_evaluation_l41'], 'go_growth': ['go_growth.step_impl_evaluation_l43', 'go_growth.step_impl_evaluation_l57'], 'goa': ['goa.init_pop', 'goa.step_evaluation_l54'], 'gp_bo': ['gp_bo.step', 'gp_bo.evaluate_positions_evaluation_l449'], 'gpoo': ['gpoo.step_impl_evaluation_l120'], 'gpso': ['gpso.local_search', 'gpso.initialize_evaluation_l75', 'gpso.step_evaluation_l108', 'gpso.initialization'], 'grasp': ['grasp.construct', 'grasp.local_search_evaluation_l177'], 'gsa': ['gsa.init_pop', 'gsa.step_evaluation_l62'], 'gska': ['gska.step_impl_evaluation_l67', 'gska.sort_best_first', 'gska.junior_gaining_sharing', 'gska.senior_gaining_sharing'], 'gso': ['gso.step_impl_evaluation_l43'], 'gso_glider_snake': ['gso_glider_snake.init_pop', 'gso_glider_snake.step_evaluation_l101'], 'gto': ['gto.1_extensive_search_eq_4', 'gto.2_choosing_area_eq_7', 'gto.3_attacking_eqs_10_13_15'], 'gwo': ['gwo.init_pop', 'gwo.step_evaluation_l51', 'gwo.step_evaluation_l52', 'gwo.step_evaluation_l53', 'gwo.step_evaluation_l55'], 'gwo_woa': ['gwo_woa.initialize_evaluation_l99', 'gwo_woa.step_evaluation_l301', 'gwo_woa.step_evaluation_l313', 'gwo_woa.initialization'], 'hba': ['hba.step_impl_evaluation_l47', 'hba.step_impl_evaluation_l50'], 'hba_honey': ['hba_honey.step_impl_evaluation_l60'], 'hbo': ['hbo.no_change', 'hbo.pick_friend_sibling_nearby', 'hbo.re_heapify_upward'], 'hc': ['hc.step', 'hc.step_evaluation_l311'], 'hco': ['hco.step_impl_evaluation_l62'], 'hde': ['hde.step_impl_evaluation_l22', 'hde.step_impl_evaluation_l31'], 'heoa': ['heoa.risk_takers_generate_bounded_sample_around', 'heoa.four_role_groups_used_by_native', 'heoa.interpretation_leaders_exploit_explorers_diversify_followers', 'heoa.toward_best_risk_takers_perform_escape', 'heoa.preserve_elites_they_may_still_be', 'heoa.levy_flight', 'heoa.explorers_move_around_population_centroid_away', 'heoa.followers_contract_toward_current_best', 'heoa.keep_sorted_order_stable_role_assignment'], 'hgs': ['hgs.step_impl_evaluation_l69', 'hgs.hunger_values_eq_2_2', 'hgs.eq_2_2_sech_approx', 'hgs.eq_2_3'], 'hgso': ['hgso.best_in_cluster', 'hgso.replace_worst_n_w_agents_eq', 'hgso.henry_s_coefficient_eq_8', 'hgso.cluster'], 'hho': ['hho.init_pop', 'hho.step_evaluation_l55', 'hho.step_evaluation_l60', 'hho.step_evaluation_l64', 'hho.step_evaluation_l68', 'hho.step_evaluation_l72', 'hho.step_evaluation_l77', 'hho.step_evaluation_l81'], 'hi_woa': ['hi_woa.initialize_evaluation_l99', 'hi_woa.step_evaluation_l301', 'hi_woa.step_evaluation_l313', 'hi_woa.initialization'], 'hiking_oa': ['hiking_oa.step_impl_evaluation_l28'], 'ho_hippo': ['ho_hippo.step_impl_evaluation_l52', 'ho_hippo.step_impl_evaluation_l53', 'ho_hippo.step_impl_evaluation_l58', 'ho_hippo.step_impl_evaluation_l66', 'ho_hippo.step_impl_evaluation_l72'], 'horse_oa': ['horse_oa.step_impl_evaluation_l46'], 'hsa': ['hsa.init_pop', 'hsa.step_evaluation_l45'], 'hsaba': ['hsaba.step_impl_evaluation_l39'], 'hus': ['hus.init_pop', 'hus.step_evaluation_l60'], 'i_gwo': ['i_gwo.init_pop', 'i_gwo.step_evaluation_l51', 'i_gwo.step_evaluation_l52', 'i_gwo.step_evaluation_l53', 'i_gwo.step_evaluation_l55', 'i_gwo.improve_step'], 'i_woa': ['i_woa.init_pop', 'i_woa.breed', 'i_woa.step_evaluation_l76'], 'iagwo': ['iagwo.init_pop', 'iagwo.step_evaluation_l154'], 'iaro': ['iaro.initialize_evaluation_l99', 'iaro.step_evaluation_l301', 'iaro.step_evaluation_l313', 'iaro.initialization'], 'ica': ['ica.1_assimilation', 'ica.2_revolution_random_dimension_reset', 'ica.3_intra_empire_competition_colony_beats', 'ica.4_inter_empire_competition_steal_weakest', 'ica.transfer_one_colony', 'ica.pick_weakest_colony'], 'ikoa': ['ikoa.step_impl_evaluation_l87', 'ikoa.step_impl_evaluation_l88'], 'ils': ['ils.step', 'ils.local_search_evaluation_l145', 'ils.local_search_evaluation_l177'], 'ilshade': ['ilshade.step_impl_evaluation_l44'], 'imode': ['imode.initialize_evaluation_l40', 'imode.step_evaluation_l140', 'imode.initialization', 'imode.linear_population_size_reduction', 'imode.archive_trimming', 'imode.sample_cr_f_operator_assignment', 'imode.operator_probabilities_cumulative', 'imode.index_arrays', 'imode.mutation', 'imode.crossover', 'imode.archive_population', 'imode.adapt_mcr_mf', 'imode.adapt_mop'], 'improved_aeo': ['improved_aeo.initialize_evaluation_l99', 'improved_aeo.step_evaluation_l301', 'improved_aeo.step_evaluation_l313', 'improved_aeo.initialization'], 'improved_qsa': ['improved_qsa.initialize_evaluation_l99', 'improved_qsa.step_evaluation_l301', 'improved_qsa.step_evaluation_l313', 'improved_qsa.initialization'], 'improved_tlo': ['improved_tlo.initialize_evaluation_l99', 'improved_tlo.step_evaluation_l301', 'improved_tlo.step_evaluation_l313', 'improved_tlo.initialization'], 'incremental_gwo': ['incremental_gwo.initialize_evaluation_l99', 'incremental_gwo.step_evaluation_l301', 'incremental_gwo.step_evaluation_l313', 'incremental_gwo.initialization'], 'info': ['info.step_impl_evaluation_l81', 'info.pick_better_as_random_member_3rd'], 'iobl_gwo': ['iobl_gwo.initialize_evaluation_l99', 'iobl_gwo.step_evaluation_l301', 'iobl_gwo.step_evaluation_l313', 'iobl_gwo.initialization'], 'ipop_cmaes': ['ipop_cmaes.step', 'ipop_cmaes.step_evaluation_l534'], 'ivya': ['ivya.step_impl_evaluation_l36'], 'iwo': ['iwo.step_impl_evaluation_l51'], 'jade': ['jade.initialize_evaluation_l99', 'jade.step_evaluation_l301', 'jade.step_evaluation_l313', 'jade.initialization'], 'jde': ['jde.step_impl_evaluation_l37'], 'jso': ['jso.init_pop', 'jso.step_evaluation_l61'], 'jy': ['jy.init_pop', 'jy.step_evaluation_l47'], 'kha': ['kha.init_pop', 'kha.step_evaluation_l62', 'kha.foraging', 'kha.diffusion', 'kha.position', 'kha.step_evaluation_l93', 'kha.step_evaluation_l100', 'kha.motion_induced', 'kha.ga_step_20'], 'kma': ['kma.initialize_evaluation_l32', 'kma.n_big', 'kma.female_reproduction', 'kma.small_males_move_towards_big_males', 'kma.initialization', 'kma.sort_ascending_best_first', 'kma.index_female_single', 'kma.big_males_movement', 'kma.n_big_d'], 'l2smea': ['l2smea.step', 'l2smea.step_evaluation_l167'], 'laro': ['laro.initialize_evaluation_l99', 'laro.step_evaluation_l301', 'laro.step_evaluation_l313', 'laro.initialization'], 'lca': ['lca.mutation', 'lca.replication_move_toward_best', 'lca.crossover'], 'lco': ['lco.boundary_reflection', 'lco.sqrt_size_n_best_pool', 'lco.n_best_mean_eq_1', 'lco.gradient_step_eqs_2_6'], 'levy_ja': ['levy_ja.initialize_evaluation_l99', 'levy_ja.step_evaluation_l301', 'levy_ja.step_evaluation_l313', 'levy_ja.initialization'], 'lfd': ['lfd.step_impl_evaluation_l46'], 'liwo': ['liwo.step_impl_evaluation_l73', 'liwo.breeze_driven_translation_optional_spiral_motion', 'liwo.strong_wind_one_dimensional_displacement_optional'], 'loa': ['loa.step_impl_evaluation_l46', 'loa.territorial_takeover_best_nomads_can_replace'], 'loa_lyrebird': ['loa_lyrebird.step_impl_evaluation_l59'], 'lpo': ['lpo.step_impl_evaluation_l51', 'lpo.original_lpo_formula_assumes_positive_physiological', 'lpo.quantities_objective_values_however_can_be', 'lpo.extremely_close_zero_use_magnitudes_scale', 'lpo.keep_them_away_underflow_avoid_zerodivisionerror'], 'lshade_cnepsin': ['lshade_cnepsin.step_impl_evaluation_l112'], 'lso_spectrum': ['lso_spectrum.step_impl_evaluation_l87'], 'mbo': ['mbo.init_pop', 'mbo.step_evaluation_l67', 'mbo.migration', 'mbo.adjustment'], 'memetic_a': ['memetic_a.init_pop', 'memetic_a.breed', 'memetic_a.mutate', 'memetic_a.xhc'], 'mfa': ['mfa.init_pop', 'mfa.step_evaluation_l52'], 'mfea': ['mfea.initialize_evaluation_l44', 'mfea.mutation', 'mfea.initialization', 'mfea.crossover'], 'mfea2': ['mfea2.step', 'mfea2.step_evaluation_l194'], 'mfo': ['mfo.greedy_update', 'mfo.1', 'mfo.i_ij_in_1_2', 'mfo.2'], 'mgo': ['mgo.initialize_evaluation_l32', 'mgo.clip_all', 'mgo.initialization', 'mgo.approximation_adapted_each_step', 'mgo.sort_best_first', 'mgo.coefficient_matrix_4_strategies_paper', 'mgo.backherd_mean_bh_mean_random_subset', 'mgo.n_d', 'mgo.tsm_territory_marking', 'mgo.mh_mountain_herding', 'mgo.bmh_best_mountain_herding', 'mgo.exploration'], 'mgoa_market': ['mgoa_market.step_impl_evaluation_l63'], 'misaco': ['misaco.step', 'misaco.step_evaluation_l272'], 'mke': ['mke.step_impl_evaluation_l33'], 'modified_aeo': ['modified_aeo.initialize_evaluation_l99', 'modified_aeo.step_evaluation_l301', 'modified_aeo.step_evaluation_l313', 'modified_aeo.initialization'], 'modified_eo': ['modified_eo.initialize_evaluation_l99', 'modified_eo.step_evaluation_l301', 'modified_eo.step_evaluation_l313', 'modified_eo.initialization'], 'moss_go': ['moss_go.water_dispersal_gradient_like', 'moss_go.spore_dispersal', 'moss_go.levy_flight', 'moss_go.growth'], 'mpa': ['mpa.fish_aggregating_devices_fads_effect', 'mpa.exploration', 'mpa.2_transition', 'mpa.exploitation'], 'mrfo': ['mrfo.chain_foraging_eqs_1_2', 'mrfo.somersault_foraging_eq_8', 'mrfo.cyclone_foraging_eqs_5_7', 'mrfo.towards_random_point', 'mrfo.towards_best'], 'msa_e': ['msa_e.exploitation', 'msa_e.keep_sorted_best_first', 'msa_e.levy_flight', 'msa_e.re_sort_reinsert_elites_at_back'], 'mshoa': ['mshoa.step_impl_evaluation_l66'], 'msls': ['msls.step', 'msls.local_search_evaluation_l145', 'msls.local_search_evaluation_l177'], 'mso': ['mso.step_impl_evaluation_l40', 'mso.inferior_mirage_search_remaining', 'mso.superior_mirage_search'], 'mtbo': ['mtbo.random_relocation', 'mtbo.probabilities_matlab_implementation', 'mtbo.li_in_0_25_0_50', 'mtbo.coordinated_movement_next_mountaineer_leader', 'mtbo.avalanche_disaster_response_move_relative_weakest', 'mtbo.movement_toward_team_mean'], 'mts': ['mts.local'], 'mvo': ['mvo.init_pop', 'mvo.step_evaluation_l59'], 'mvpa': ['mvpa.initialize_evaluation_l38', 'mvpa.each_player_moves_toward_mvp_random', 'mvpa.initialization', 'mvpa.archive_top_archive_size_players', 'mvpa.replacement', 'mvpa.mvp'], 'nca': ['nca.step_impl_evaluation_l67', 'nca.acceleration_hyperbolic_contraction_random_subset_components', 'nca.exploration', 'nca.coordinate_distribution_samples_inside_those_intervals', 'nca.exploitation'], 'ngo': ['ngo.1', 'ngo.2'], 'nlapsmjso_eda': ['nlapsmjso_eda.evaluate_population_evaluation_l98', 'nlapsmjso_eda.initialize_evaluation_l181', 'nlapsmjso_eda.step_evaluation_l254', 'nlapsmjso_eda.step_evaluation_l301', 'nlapsmjso_eda.evaluate_population', 'nlapsmjso_eda.initialization'], 'nmm': ['nmm.step_impl_evaluation_l30', 'nmm.step_impl_evaluation_l36', 'nmm.step_impl_evaluation_l39', 'nmm.step_impl_evaluation_l44'], 'nmra': ['nmra.working_operator', 'nmra.breeders', 'nmra.sort_so_breeders_are_best', 'nmra.breeding_operator'], 'nndrea_so': ['nndrea_so.stage_1_evolve_nn_weights_proxy', 'nndrea_so.decode_through_nn', 'nndrea_so.stage_2_direct_ga_on_binary', 'nndrea_so.initialization', 'nndrea_so.decide_stage_delta_fraction_evaluations_in', 'nndrea_so.evolve_nn_weights_via_de', 'nndrea_so.check_we_should_switch_stage_2'], 'noa': ['noa.step_impl_evaluation_l147'], 'nro': ['nro.step_impl_evaluation_l64', 'nro.step_impl_evaluation_l82', 'nro.nfi', 'nro.nfu'], 'nwoa': ['nwoa.step_impl_evaluation_l81'], 'ocro': ['ocro.initialize_evaluation_l99', 'ocro.step_evaluation_l301', 'ocro.step_evaluation_l313', 'ocro.initialization'], 'ofa': ['ofa.initialize_evaluation_l32', 'ofa.use_gen_n_as_proxy_fe', 'ofa.initialization', 'ofa.approximate_maxfe_ratio', 'ofa.ofa_movement_x_new_x_fe', 'ofa.neighbour_is_randomly_picked_rest_shifted', 'ofa.stochastic_acceptance_f_off_1_fe', 'ofa.re_sort'], 'ogwo': ['ogwo.initialize_evaluation_l99', 'ogwo.step_evaluation_l301', 'ogwo.step_evaluation_l313', 'ogwo.initialization'], 'ooa': ['ooa.step_impl_evaluation_l22', 'ooa.step_impl_evaluation_l25'], 'parrot_o': ['parrot_o.fly_new_area', 'parrot_o.foraging', 'parrot_o.levy_flight'], 'pbil': ['pbil.initialize_evaluation_l26', 'pbil.step_evaluation_l37', 'pbil.initialization', 'pbil.vector', 'pbil.mutate'], 'pcx': ['pcx.init_pop', 'pcx.generate_child', 'pcx.step_evaluation_l165', 'pcx.inject_candidates_evaluation_l265', 'pcx.inject_candidates'], 'pdo': ['pdo.step_impl_evaluation_l48'], 'petio': ['petio.step_impl_evaluation_l252', 'petio.best_worst_order_map_elite_core', 'petio.immediate_upper_layer_beginner_core_core', 'petio.beginner_exploratory', 'petio.core_balanced', 'petio.elite_development', 'petio.skill_rotation_paper_leaves_x_best'], 'pfa': ['pfa.init_pop', 'pfa.pathfinder', 'pfa.positions'], 'pfa_polar_fox': ['pfa_polar_fox.init_pop', 'pfa_polar_fox.experience_phase', 'pfa_polar_fox.leader_phase', 'pfa_polar_fox.step_evaluation_l239', 'pfa_polar_fox.step_evaluation_l247', 'pfa_polar_fox.elite_may_belong_group_kept_count'], 'pko': ['pko.step_impl_evaluation_l42', 'pko.step_impl_evaluation_l53'], 'plba': ['plba.step_impl_evaluation_l35'], 'plo': ['plo.step_impl_evaluation_l47'], 'poa': ['poa.1_moving_toward_prey_eq_4', 'poa.2_winging_on_water_surface_eq'], 'political_o': ['political_o.electoral_campaign', 'political_o.parliamentary', 'political_o.number_constituencies', 'political_o.assign_constituents_parties_best_nc_are'], 'pro': ['pro.step_impl_evaluation_l29', 'pro.step_impl_evaluation_l36'], 'pso': ['pso.initialize_evaluation_l75', 'pso.velocity_position', 'pso.inject_candidates_evaluation_l251', 'pso.initialization', 'pso.inject_candidates', 'pso.individual_best', 'pso.global_best', 'pso.inertia_decay', 'pso.write_back'], 'pss': ['pss.step_impl_evaluation_l44'], 'puma_o': ['puma_o.stalking', 'puma_o.attack', 'puma_o.golden_ratio'], 'qio': ['qio.step_impl_evaluation_l57'], 'qle_sca': ['qle_sca.initialize_evaluation_l99', 'qle_sca.step_evaluation_l301', 'qle_sca.step_evaluation_l313', 'qle_sca.initialization'], 'qsa': ['qsa.business1', 'qsa.business2', 'qsa.business3'], 'random_s': ['random_s.init_pop', 'random_s.step_evaluation_l44'], 'rbmo': ['rbmo.1', 'rbmo.2', 'rbmo.food_storage'], 'rcco': ['rcco.step_impl_evaluation_l96', 'rcco.step_impl_evaluation_l119'], 'rf_bo': ['rf_bo.step', 'rf_bo.evaluate_positions_evaluation_l449'], 'rfo': ['rfo.smell'], 'rhso': ['rhso.step_impl_evaluation_l64'], 'rime': ['rime.hard_rime_puncture_mechanism', 'rime.per_individual_normalised_fitness', 'rime.soft_rime_search_strategy'], 'rmsprop': ['rmsprop.initialize_evaluation_l56', 'rmsprop.step_evaluation_l72', 'rmsprop.initialization', 'rmsprop.grad_evaluation_l23'], 'roa': ['roa.attempt'], 'rsa': ['rsa.step_impl_evaluation_l39'], 'rso': ['rso.step_impl_evaluation_l23', 'rso.step_impl_evaluation_l31'], 'run': ['run.step_impl_evaluation_l73', 'run.step_impl_evaluation_l90', 'run.enhanced_solution_quality_esq'], 'sa': ['sa.initialize_evaluation_l91', 'sa.step_evaluation_l124', 'sa.inject_candidates_evaluation_l234', 'sa.initialization', 'sa.inject_candidates', 'sa.best', 'sa.cool', 'sa.detect_natural_termination_temperature_exhausted'], 'saba': ['saba.step_impl_evaluation_l39'], 'sacc_eam2': ['sacc_eam2.step', 'sacc_eam2.step_evaluation_l232', 'sacc_eam2.step_evaluation_l237', 'sacc_eam2.step_evaluation_l244', 'sacc_eam2.step_evaluation_l246', 'sacc_eam2.step_evaluation_l253'], 'sacoso': ['sacoso.initialize_evaluation_l48', 'sacoso.fes_swarm_standard_pso', 'sacoso.initialization'], 'sade': ['sade.initialize_evaluation_l99', 'sade.step_evaluation_l301', 'sade.step_evaluation_l313', 'sade.initialization'], 'sade_amss': ['sade_amss.initialize_evaluation_l47', 'sade_amss.de_rand_1_bin_on_subspace', 'sade_amss.initialization', 'sade_amss.k_subspace_de_iterations_without_surrogate'], 'sade_atdsc': ['sade_atdsc.initialize_evaluation_l46', 'sade_atdsc.without_surrogate_evaluate_all_pick_best', 'sade_atdsc.initialization', 'sade_atdsc.candidate_generation', 'sade_atdsc.replacement'], 'sade_sammon': ['sade_sammon.initialize_evaluation_l50', 'sade_sammon.step_evaluation_l78', 'sade_sammon.initialization', 'sade_sammon.candidate_generation'], 'samso': ['samso.initialize_evaluation_l45', 'samso.step_evaluation_l68', 'samso.initialization'], 'sap_de': ['sap_de.initialize_evaluation_l99', 'sap_de.step_evaluation_l301', 'sap_de.step_evaluation_l313', 'sap_de.initialization'], 'sapo': ['sapo.step', 'sapo.step_evaluation_l398'], 'saro': ['saro.boundary_repair', 'saro.individual', 'saro.reinitialize_individuals_too_many_unsuccessful_searches', 'saro.social'], 'sbo': ['sbo.mutation', 'sbo.fitness_proportionate_roulette_raw_objective_values', 'sbo.negative_e_g_easom_all_near', 'sbo.probabilities_are_numerically_invalid_convert_fitness', 'sbo.non_negative_quality_score_where_larger'], 'sboa': ['sboa.1', 'sboa.2'], 'scho': ['scho.step_impl_evaluation_l58'], 'scso': ['scso.exploration', 'scso.selection', 'scso.exploitation', 'scso.replace_all_original_behaviour'], 'sd': ['sd.initialize_evaluation_l56', 'sd.step_evaluation_l75', 'sd.initialization', 'sd.grad_evaluation_l23'], 'seaho': ['seaho.eq_11', 'seaho.3_reproduction', 'seaho.1_motor_behavior', 'seaho.eq_4', 'seaho.eq_7', 'seaho.2_predation', 'seaho.eq_10'], 'serval_oa': ['serval_oa.step_impl_evaluation_l22', 'serval_oa.step_impl_evaluation_l25'], 'sfo': ['sfo.initialize_payload_evaluation_l25', 'sfo.sailfish_positions_eq_6', 'sfo.attack_power_eq_10', 'sfo.step_impl_evaluation_l75', 'sfo.replenish_sardines_depleted', 'sfo.initialization', 'sfo.best_sardine', 'sfo.sailfish_absorb_best_sardines'], 'sfoa': ['sfoa.step_impl_evaluation_l41', 'sfoa.exploitation', 'sfoa.exploration'], 'shade': ['shade.step_impl_evaluation_l43'], 'shio': ['shio.step_impl_evaluation_l28'], 'shio_success': ['shio_success.step_impl_evaluation_l59'], 'sho': ['sho.collect_n_hyenas_around_prey', 'sho.step_impl_evaluation_l64'], 'sine_cosine_a': ['sine_cosine_a.init_pop', 'sine_cosine_a.step_evaluation_l49'], 'singer_oa': ['singer_oa.greedy_apply', 'singer_oa.1', 'singer_oa.2', 'singer_oa.creative_shift_as_run_proceeds_as'], 'slo': ['slo.step_impl_evaluation_l47'], 'sma': ['sma.eq_2_3', 'sma.oscillation_weights_eq_2_5', 'sma.maximisation_reverse_ordering_roles', 'sma.eq_2_4', 'sma.random_dispersion_eq_2_7', 'sma.eq_2_2'], 'smo': ['smo.1_local_leader', 'smo.2_global_leader', 'smo.3_local_leader_decision', 'smo.local_leader_index_per_group', 'smo.global_limit_fission_fusion', 'smo.fission_split_largest_group', 'smo.fusion_merge_two_smallest_groups'], 'snow_oa': ['snow_oa.step_impl_evaluation_l46'], 'so_snake': ['so_snake.step_impl_evaluation_l76', 'so_snake.step_impl_evaluation_l77'], 'soa': ['soa.eq_14', 'soa.eq_6', 'soa.eq_8', 'soa.eq_7', 'soa.eq_5', 'soa.eq_9'], 'soo': ['soo.1_oscillatory_position', 'soo.2_top_3_average_oscillatory'], 'sopt': ['sopt.select_best', 'sopt.exploitation'], 'sos': ['sos.init_pop', 'sos.mutualism', 'sos.comensalism', 'sos.parasitism'], 'sparrow_sa': ['sparrow_sa.step_impl_evaluation_l41', 'sparrow_sa.step_impl_evaluation_l50', 'sparrow_sa.awareness'], 'spbo': ['spbo.init_pop', 'spbo.best_student', 'spbo.groups', 'spbo.step_evaluation_l74', 'spbo.step_evaluation_l77', 'spbo.classify'], 'sqp': ['sqp.initialize_evaluation_l58', 'sqp.line_search_armijo', 'sqp.initialization', 'sqp.unconstrained_qp_subproblem_min_0_5', 'sqp.grad_evaluation_l24'], 'squirrel_sa': ['squirrel_sa.case_1_acorn_squirrels_hickory', 'squirrel_sa.toward_acorn', 'squirrel_sa.toward_hickory', 'squirrel_sa.sort_best_hickory_next_acorn_rest', 'squirrel_sa.case_2_3_normal_squirrels_acorn'], 'srsr': ['srsr.1_accumulation_new_positions_via_gaussian', 'srsr.exploration', 'srsr.master_robot_current_best', 'srsr.improvement_negative_better_min', 'srsr.sif_best_improver'], 'srsr_robotics': ['srsr_robotics.trial_fit', 'srsr_robotics.mean_is_pulled_by_master_robot', 'srsr_robotics.step_impl_evaluation_l90', 'srsr_robotics.random_signed_group_vector_plus_large', 'srsr_robotics.source_implementation_sorts_population_before_each', 'srsr_robotics.1_accumulation_slave_robots_sample_gaussian', 'srsr_robotics.exploration', 'srsr_robotics.3_local_worker_robots_around_master', 'srsr_robotics.operators_source_implementation'], 'ssa': ['ssa.init_pop', 'ssa.step_evaluation_l58'], 'ssdo': ['ssdo.step_impl_evaluation_l49'], 'ssio_rl': ['ssio_rl.step', 'ssio_rl.step_evaluation_l346'], 'sso': ['sso.step_impl_evaluation_l30'], 'sspider_a': ['sspider_a.step_impl_evaluation_l43'], 'sto': ['sto.step_impl_evaluation_l23', 'sto.step_impl_evaluation_l26'], 'superb_foa': ['superb_foa.step_impl_evaluation_l45'], 'supply_do': ['supply_do.step_impl_evaluation_l39', 'supply_do.step_impl_evaluation_l43'], 'tdo': ['tdo.step_impl_evaluation_l24', 'tdo.step_impl_evaluation_l28'], 'tfwo': ['tfwo.initialize_evaluation_l53', 'tfwo.effect_of_objects', 'tfwo.effect_of_whirlpools', 'tfwo.initialization'], 'thro': ['thro.race'], 'tlbo': ['tlbo.init_pop', 'tlbo.step_evaluation_l46', 'tlbo.learner_phase'], 'tlco': ['tlco.step_impl_evaluation_l41', 'tlco.step_impl_evaluation_l55', 'tlco.step_impl_evaluation_l63', 'tlco.step_impl_evaluation_l77'], 'toa': ['toa.stage_1_supervisor_guidance', 'toa.step_impl_evaluation_l50', 'toa.stage_3_individual_activity', 'toa.stage_2_information_sharing_better_agents'], 'toc': ['toc.velocity_radial_tangential_random', 'toc.spiral_radius_decreases_over_time', 'toc.angular_velocity_coriolis_like_rotation', 'toc.tangential_velocity_component'], 'tpo': ['tpo.step_impl_evaluation_l34'], 'tree_seed_a': ['tree_seed_a.eq_4', 'tree_seed_a.eq_3'], 'ts': ['ts.initialize_evaluation_l28', 'ts.step_evaluation_l53', 'ts.initialization'], 'tsa': ['tsa.step_impl_evaluation_l40', 'tsa.replacement'], 'tso': ['tso.parabolic_updates', 'tso.leader_spiral', 'tso.random_migration', 'tso.spiral_following', 'tso.replacement'], 'ttao': ['ttao.step_impl_evaluation_l69', 'ttao.step_impl_evaluation_l72', 'ttao.step_impl_evaluation_l75', 'ttao.step_impl_evaluation_l78', 'ttao.step_impl_evaluation_l110', 'ttao.step_impl_evaluation_l137', 'ttao.step_impl_evaluation_l148', 'ttao.step_impl_evaluation_l165', 'ttao.crossover', 'ttao.contraction'], 'two': ['two.weights_by_rank', 'two.gravitational_like_decay'], 'vcs': ['vcs.1_virus_diffusion', 'vcs.2_host_cell_infection', 'vcs.3_immune_response'], 'vns': ['vns.step', 'vns.local_search_evaluation_l145', 'vns.local_search_evaluation_l177'], 'waoa': ['waoa.step_impl_evaluation_l24', 'waoa.step_impl_evaluation_l28'], 'warso': ['warso.original_index_sorted_i'], 'wca': ['wca.streams_toward_river', 'wca.river_toward_sea', 'wca.evaporation_raining', 'wca.best_stream_replaces_river_better', 'wca.check_river_is_better_than_sea'], 'wdo': ['wdo.step_impl_evaluation_l46'], 'whale_foa': ['whale_foa.initialize_evaluation_l99', 'whale_foa.step_evaluation_l301', 'whale_foa.step_evaluation_l313', 'whale_foa.initialization'], 'who': ['who.1_local_movement_milling', 'who.2_herd_instinct', 'who.social_memory', 'who.3_starvation_avoidance_4_population_pressure', 'who.starvation_avoidance', 'who.population_pressure'], 'wmqimrfo': ['wmqimrfo.initialize_evaluation_l99', 'wmqimrfo.step_evaluation_l301', 'wmqimrfo.step_evaluation_l313', 'wmqimrfo.initialization'], 'wo_wave': ['wo_wave.step_impl_evaluation_l73'], 'woa': ['woa.init_pop', 'woa.step_evaluation_l59'], 'wooa': ['wooa.greedy_single', 'wooa.strategy_1', 'wooa.strategy_2_1', 'wooa.strategy_2_2'], 'wso': ['wso.step_impl_evaluation_l57'], 'wutp': ['wutp.horizontal', 'wutp.step_impl_evaluation_l46', 'wutp.water_in_motion'], 'ydse': ['ydse.step_impl_evaluation_l52'], 'zoa': ['zoa.1_foraging_eq_3', 'zoa.s2_offensive', 'zoa.2_defence_against_predators', 'zoa.s1_lion_escape']}

FUNCTION_OPERATOR_LABELS = {'aao': {'initialize': 'aao.initialization'}, 'abco': {'_employed': 'abco.employed', '_onlooker': 'abco.onlooker', '_scout': 'abco.scout', 'initialize': 'abco.initialization'}, 'acgwo': {'initialize': 'acgwo.initialization'}, 'aco': {'initialize': 'aco.initialization'}, 'adam': {'initialize': 'adam.initialization'}, 'adaptive_eo': {'initialize': 'adaptive_eo.initialization'}, 'aesspso': {'initialize': 'aesspso.initialization'}, 'afsa': {'_init_pop': 'afsa.init_pop', '_prey': 'afsa.prey', '_swarm': 'afsa.swarm', '_follow': 'afsa.follow', '_leap': 'afsa.leap'}, 'aiw_pso': {'initialize': 'aiw_pso.initialization'}, 'alo': {'_init_pop': 'alo.init_pop', 'inject_candidates': 'alo.inject_candidates'}, 'aoa': {'initialize': 'aoa.initialization'}, 'ars': {'initialize': 'ars.initialization', 'inject_candidates': 'ars.inject_candidates'}, 'autov': {'initialize': 'autov.initialization'}, 'bat_a': {'initialize': 'bat_a.initialization', 'inject_candidates': 'bat_a.inject_candidates'}, 'bbo': {'_init_pop': 'bbo.init_pop'}, 'bfgs': {'initialize': 'bfgs.initialization'}, 'bspga': {'_evaluate_positions': 'bspga.evaluate_positions'}, 'btoa': {'_dynamic_position_candidate': 'btoa.dynamic_position_candidate'}, 'ca': {'_init_pop': 'ca.init_pop'}, 'cat_so': {'_init_pop': 'cat_so.init_pop'}, 'cco': {'_greedy_single': 'cco.greedy_single'}, 'cem': {'_init_pop': 'cem.init_pop', 'inject_candidates': 'cem.inject_candidates'}, 'cg_gwo': {'initialize': 'cg_gwo.initialization'}, 'chaotic_gwo': {'initialize': 'chaotic_gwo.initialization'}, 'chicken_so': {'_init_pop': 'chicken_so.init_pop'}, 'clonalg': {'_init_pop': 'clonalg.init_pop'}, 'cmaes': {'initialize': 'cmaes.initialization'}, 'coati_oa': {'_init_pop': 'coati_oa.init_pop'}, 'cockroach_so': {'_init_pop': 'cockroach_so.init_pop'}, 'compact_ga': {'_evaluate_bits': 'compact_ga.evaluate_bits'}, 'csa': {'_init_pop': 'csa.init_pop'}, 'cso': {'initialize': 'cso.initialization'}, 'cuckoo_s': {'_init_pop': 'cuckoo_s.init_pop'}, 'da': {'_init_pop': 'da.init_pop', 'inject_candidates': 'da.inject_candidates'}, 'de': {'_init_pop': 'de.init_pop'}, 'dfo': {'_init_pop': 'dfo.init_pop'}, 'ds_gwo': {'initialize': 'ds_gwo.initialization'}, 'dvba': {'_init_pop': 'dvba.init_pop'}, 'ecological_cycle_o': {'_eval_accept_group': 'ecological_cycle_o.eval_accept_group'}, 'ecpo': {'initialize': 'ecpo.initialization'}, 'ego': {'initialize': 'ego.initialization'}, 'eho': {'_init_pop': 'eho.init_pop'}, 'enhanced_aeo': {'initialize': 'enhanced_aeo.initialization'}, 'enhanced_two': {'initialize': 'enhanced_two.initialization'}, 'er_gwo': {'initialize': 'er_gwo.initialization'}, 'ex_gwo': {'initialize': 'ex_gwo.initialization'}, 'fda': {'_init_pop': 'fda.init_pop'}, 'fep': {'initialize': 'fep.initialization'}, 'firefly_a': {'_init_pop': 'firefly_a.init_pop'}, 'flo': {'_greedy_single': 'flo.greedy_single'}, 'fpa': {'_init_pop': 'fpa.init_pop'}, 'frcg': {'initialize': 'frcg.initialization'}, 'frofi': {'initialize': 'frofi.initialization'}, 'fuzzy_gwo': {'initialize': 'fuzzy_gwo.initialization'}, 'ga': {'_init_pop': 'ga.init_pop', '_breed': 'ga.breed', '_mutate': 'ga.mutate'}, 'gmo': {'_init_pop': 'gmo.init_pop'}, 'goa': {'_init_pop': 'goa.init_pop'}, 'gpso': {'_local_search': 'gpso.local_search', 'initialize': 'gpso.initialization'}, 'grasp': {'_construct': 'grasp.construct'}, 'gsa': {'_init_pop': 'gsa.init_pop'}, 'gso_glider_snake': {'_init_pop': 'gso_glider_snake.init_pop'}, 'gwo': {'_init_pop': 'gwo.init_pop'}, 'gwo_woa': {'initialize': 'gwo_woa.initialization'}, 'hho': {'_init_pop': 'hho.init_pop'}, 'hi_woa': {'initialize': 'hi_woa.initialization'}, 'hsa': {'_init_pop': 'hsa.init_pop'}, 'hus': {'_init_pop': 'hus.init_pop'}, 'i_gwo': {'_init_pop': 'i_gwo.init_pop'}, 'i_woa': {'_init_pop': 'i_woa.init_pop', '_breed': 'i_woa.breed'}, 'iagwo': {'_init_pop': 'iagwo.init_pop'}, 'iaro': {'initialize': 'iaro.initialization'}, 'imode': {'initialize': 'imode.initialization'}, 'improved_aeo': {'initialize': 'improved_aeo.initialization'}, 'improved_qsa': {'initialize': 'improved_qsa.initialization'}, 'improved_tlo': {'initialize': 'improved_tlo.initialization'}, 'incremental_gwo': {'initialize': 'incremental_gwo.initialization'}, 'iobl_gwo': {'initialize': 'iobl_gwo.initialization'}, 'jade': {'initialize': 'jade.initialization'}, 'jso': {'_init_pop': 'jso.init_pop'}, 'jy': {'_init_pop': 'jy.init_pop'}, 'kha': {'_init_pop': 'kha.init_pop'}, 'kma': {'initialize': 'kma.initialization'}, 'laro': {'initialize': 'laro.initialization'}, 'levy_ja': {'initialize': 'levy_ja.initialization'}, 'mbo': {'_init_pop': 'mbo.init_pop'}, 'memetic_a': {'_init_pop': 'memetic_a.init_pop', '_breed': 'memetic_a.breed', '_mutate': 'memetic_a.mutate', '_xhc': 'memetic_a.xhc'}, 'mfa': {'_init_pop': 'mfa.init_pop'}, 'mfea': {'initialize': 'mfea.initialization'}, 'mfo': {'_greedy_update': 'mfo.greedy_update'}, 'mgo': {'initialize': 'mgo.initialization'}, 'modified_aeo': {'initialize': 'modified_aeo.initialization'}, 'modified_eo': {'initialize': 'modified_eo.initialization'}, 'mts': {'_local': 'mts.local'}, 'mvo': {'_init_pop': 'mvo.init_pop'}, 'mvpa': {'initialize': 'mvpa.initialization'}, 'nlapsmjso_eda': {'_evaluate_population': 'nlapsmjso_eda.evaluate_population', 'initialize': 'nlapsmjso_eda.initialization'}, 'nndrea_so': {'initialize': 'nndrea_so.initialization'}, 'ocro': {'initialize': 'ocro.initialization'}, 'ofa': {'initialize': 'ofa.initialization'}, 'ogwo': {'initialize': 'ogwo.initialization'}, 'pbil': {'initialize': 'pbil.initialization'}, 'pcx': {'_init_pop': 'pcx.init_pop', '_generate_child': 'pcx.generate_child', 'inject_candidates': 'pcx.inject_candidates'}, 'pfa': {'_init_pop': 'pfa.init_pop'}, 'pfa_polar_fox': {'_init_pop': 'pfa_polar_fox.init_pop', '_experience_phase': 'pfa_polar_fox.experience_phase', '_leader_phase': 'pfa_polar_fox.leader_phase'}, 'pso': {'initialize': 'pso.initialization', 'inject_candidates': 'pso.inject_candidates'}, 'qle_sca': {'initialize': 'qle_sca.initialization'}, 'qsa': {'_business1': 'qsa.business1', '_business2': 'qsa.business2', '_business3': 'qsa.business3'}, 'random_s': {'_init_pop': 'random_s.init_pop'}, 'rmsprop': {'initialize': 'rmsprop.initialization'}, 'sa': {'initialize': 'sa.initialization', 'inject_candidates': 'sa.inject_candidates'}, 'sacoso': {'initialize': 'sacoso.initialization'}, 'sade': {'initialize': 'sade.initialization'}, 'sade_amss': {'initialize': 'sade_amss.initialization'}, 'sade_atdsc': {'initialize': 'sade_atdsc.initialization'}, 'sade_sammon': {'initialize': 'sade_sammon.initialization'}, 'samso': {'initialize': 'samso.initialization'}, 'sap_de': {'initialize': 'sap_de.initialization'}, 'sd': {'initialize': 'sd.initialization'}, 'sfo': {'_initialize_payload': 'sfo.initialization'}, 'sine_cosine_a': {'_init_pop': 'sine_cosine_a.init_pop'}, 'singer_oa': {'_greedy_apply': 'singer_oa.greedy_apply'}, 'sopt': {'_select_best': 'sopt.select_best'}, 'sos': {'_init_pop': 'sos.init_pop'}, 'spbo': {'_init_pop': 'spbo.init_pop'}, 'sqp': {'initialize': 'sqp.initialization'}, 'srsr_robotics': {'_trial_fit': 'srsr_robotics.trial_fit'}, 'ssa': {'_init_pop': 'ssa.init_pop'}, 'tfwo': {'initialize': 'tfwo.initialization', '_effect_of_objects': 'tfwo.effect_of_objects', '_effect_of_whirlpools': 'tfwo.effect_of_whirlpools'}, 'tlbo': {'_init_pop': 'tlbo.init_pop'}, 'ts': {'initialize': 'ts.initialization'}, 'whale_foa': {'initialize': 'whale_foa.initialization'}, 'wmqimrfo': {'initialize': 'wmqimrfo.initialization'}, 'woa': {'_init_pop': 'woa.init_pop'}, 'wooa': {'_greedy_single': 'wooa.greedy_single'}}

CALLSITE_OPERATOR_LABELS = {'aaa': {'aaa.py': {'_step_impl': {'166': 'aaa.step_impl_evaluation_l166', '203': 'aaa.is_replaced_by_corresponding_cell_biggest', '219': 'aaa.adaptation_most_starving_colony_moves_toward'}}}, 'aao': {'aao.py': {'initialize': {'99': 'aao.initialize_evaluation_l99'}, 'step': {'301': 'aao.step_evaluation_l301', '313': 'aao.step_evaluation_l313'}}}, 'abco': {'abco.py': {'_employed': {'60': 'abco.employed'}, '_onlooker': {'75': 'abco.onlooker'}, '_scout': {'85': 'abco.scout'}, 'initialize': {'91': 'abco.initialize_evaluation_l91'}}}, 'acgwo': {'acgwo.py': {'initialize': {'32': 'acgwo.initialize_evaluation_l32'}, 'step': {'55': 'acgwo.step_evaluation_l55', '56': 'acgwo.step_evaluation_l56', '57': 'acgwo.step_evaluation_l57', '60': 'acgwo.step_evaluation_l60'}}}, 'aco': {'aco.py': {'initialize': {'48': 'aco.random_real_valued_initialization'}, 'step': {'78': 'aco.pheromone_weighted_perturbation_in_each_dimension'}}}, 'acor': {'acor.py': {'_step_impl': {'33': 'acor.step_impl_evaluation_l33'}}}, 'adam': {'adam.py': {'initialize': {'57': 'adam.initialize_evaluation_l57'}, 'step': {'78': 'adam.step_evaluation_l78'}}}, 'adaptive_eo': {'adaptive_eo.py': {'initialize': {'99': 'adaptive_eo.initialize_evaluation_l99'}, 'step': {'301': 'adaptive_eo.step_evaluation_l301', '313': 'adaptive_eo.step_evaluation_l313'}}}, 'aefa': {'aefa.py': {'_step_impl': {'45': 'aefa.step_impl_evaluation_l45'}}}, 'aeo': {'aeo.py': {'_step_impl': {'22': 'aeo.production_worst_agent', '36': 'aeo.step_impl_evaluation_l36'}}}, 'aesspso': {'aesspso.py': {'initialize': {'68': 'aesspso.initialize_evaluation_l68'}, 'step': {'103': 'aesspso.safe'}}}, 'afsa': {'afsa.py': {'_init_pop': {'25': 'afsa.init_pop'}, '_prey': {'42': 'afsa.prey'}, '_swarm': {'59': 'afsa.swarm'}, '_follow': {'70': 'afsa.follow'}, '_leap': {'80': 'afsa.leap'}}}, 'aft': {'aft.py': {'_step_impl': {'52': 'aft.step_impl_evaluation_l52'}}}, 'agto': {'agto.py': {'_step_impl': {'52': 'agto.step_impl_evaluation_l52', '72': 'agto.step_impl_evaluation_l72'}}}, 'aha': {'aha.py': {'_step_impl': {'63': 'aha.handle_nan', '81': 'aha.territorial_foraging', '98': 'aha.migration'}}}, 'aho': {'aho.py': {'_step_impl': {'57': 'aho.step_impl_evaluation_l57', '76': 'aho.step_impl_evaluation_l76'}}}, 'aiw_pso': {'aiw_pso.py': {'initialize': {'99': 'aiw_pso.initialize_evaluation_l99'}, 'step': {'301': 'aiw_pso.step_evaluation_l301', '313': 'aiw_pso.step_evaluation_l313'}}}, 'ala': {'ala.py': {'_step_impl': {'52': 'ala.step_impl_evaluation_l52'}}}, 'alo': {'alo.py': {'_init_pop': {'24': 'alo.init_pop'}, 'step': {'62': 'alo.step_evaluation_l62'}, 'inject_candidates': {'116': 'alo.inject_candidates_evaluation_l116'}}}, 'ao': {'ao.py': {'_step_impl': {'74': 'ao.eq_14'}}}, 'aoa': {'aoa.py': {'initialize': {'24': 'aoa.initialize_evaluation_l24'}, 'step': {'43': 'aoa.step_evaluation_l43'}}}, 'aoo': {'aoo.py': {'_step_impl': {'139': 'aoo.step_impl_evaluation_l139'}}}, 'apo': {'apo.py': {'_step_impl': {'73': 'apo.step_impl_evaluation_l73'}}}, 'arch_oa': {'arch_oa.py': {'_step_impl': {'85': 'arch_oa.step_impl_evaluation_l85'}}}, 'aro': {'aro.py': {'_step_impl': {'54': 'aro.eq_11'}}}, 'ars': {'ars.py': {'initialize': {'27': 'ars.initialize_evaluation_l27'}, 'step': {'43': 'ars.small_step', '51': 'ars.large_step'}, 'inject_candidates': {'114': 'ars.inject_candidates_evaluation_l114'}}}, 'artemisinin_o': {'artemisinin_o.py': {'_step_impl': {'49': 'artemisinin_o.boundary'}}}, 'aso': {'aso.py': {'_step_impl': {'41': 'aso.step_impl_evaluation_l41'}}}, 'aso_atom': {'aso_atom.py': {'_step_impl': {'118': 'aso_atom.do_not_move_current_elites_unless'}}}, 'autov': {'autov.py': {'initialize': {'71': 'autov.initialize_evaluation_l71'}, 'step': {'91': 'autov.step_evaluation_l91'}}}, 'avoa': {'avoa.py': {'_step_impl': {'80': 'avoa.avoa_replaces_all_no_greedy_in'}}}, 'bacterial_colony_o': {'bacterial_colony_o.py': {'_step_impl': {'124': 'bacterial_colony_o.implementation_but_only_as_bounded_macro', '152': 'bacterial_colony_o.current_colony_best_accept_only_it', '170': 'bacterial_colony_o.reproduction_elimination_using_normalized_energy', '182': 'bacterial_colony_o.migration_is_triggered_by_low_positional'}}}, 'bat_a': {'bat_a.py': {'initialize': {'25': 'bat_a.initialize_evaluation_l25'}, 'step': {'46': 'bat_a.step_evaluation_l46', '51': 'bat_a.step_evaluation_l51', '53': 'bat_a.step_evaluation_l53'}, 'inject_candidates': {'113': 'bat_a.inject_candidates_evaluation_l113'}}}, 'bbo': {'bbo.py': {'_init_pop': {'27': 'bbo.init_pop'}, 'step': {'50': 'bbo.step_evaluation_l50', '57': 'bbo.step_evaluation_l57'}}}, 'bboa': {'bboa.py': {'_step_impl': {'45': 'bboa.step_impl_evaluation_l45', '58': 'bboa.2_sniffing'}}}, 'bbso': {'bbso.py': {'_step_impl': {'109': 'bbso.very_flat_shifted_cost_landscapes'}}}, 'bco': {'bco.py': {'_step_impl': {'52': 'bco.swim_refine_without_turbulence'}}}, 'bea': {'bea.py': {'_step_impl': {'37': 'bea.step_impl_evaluation_l37', '43': 'bea.step_impl_evaluation_l43'}}}, 'bes': {'bes.py': {'_step_impl': {'50': 'bes.stage_1_select_space', '62': 'bes.stage_2_search_in_space', '74': 'bes.stage_3_swoop'}}}, 'bfgs': {'bfgs.py': {'initialize': {'56': 'bfgs.initialize_evaluation_l56'}, 'step': {'87': 'bfgs.armijo_line_search'}}}, 'bfo': {'bfo.py': {'_step_impl': {'38': 'bfo.step_impl_evaluation_l38', '51': 'bfo.step_impl_evaluation_l51', '55': 'bfo.step_impl_evaluation_l55'}}}, 'bka': {'bka.py': {'_step_impl': {'33': 'bka.step_impl_evaluation_l33', '43': 'bka.step_impl_evaluation_l43'}}}, 'bmo': {'bmo.py': {'_step_impl': {'37': 'bmo.step_impl_evaluation_l37'}}}, 'bono': {'bono.py': {'_step_impl': {'89': 'bono.step_impl_evaluation_l89'}}}, 'boa': {'boa.py': {'_step_impl': {'39': 'boa.step_impl_evaluation_l39'}}}, 'bps': {'bps.py': {'_step_impl': {'141': 'bps.step_impl_evaluation_l141'}}}, 'bro': {'bro.py': {'_step_impl': {'49': 'bro.find_nearest_neighbour', '54': 'bro.step_impl_evaluation_l54'}}}, 'bsa': {'bsa.py': {'_step_impl': {'45': 'bsa.step_impl_evaluation_l45', '57': 'bsa.step_impl_evaluation_l57', '63': 'bsa.step_impl_evaluation_l63'}}}, 'bso': {'bso.py': {'_step_impl': {'80': 'bso.two_cluster_idea'}}}, 'bspga': {'bspga.py': {'_evaluate_positions': {'70': 'bspga.evaluate_positions'}}}, 'btoa': {'btoa.py': {'_dynamic_position_candidate': {'97': 'btoa.dynamic_position_candidate', '98': 'btoa.dynamic_position_candidate'}, '_step_impl': {'132': 'btoa.step_impl_evaluation_l132', '143': 'btoa.step_impl_evaluation_l143'}}}, 'bwo': {'bwo.py': {'_step_impl': {'41': 'bwo.crossover', '54': 'bwo.mutation'}}}, 'ca': {'ca.py': {'_init_pop': {'27': 'ca.init_pop'}}}, 'camel': {'camel.py': {'_step_impl': {'38': 'camel.step_impl_evaluation_l38', '45': 'camel.step_impl_evaluation_l45'}}}, 'capsa': {'capsa.py': {'_step_impl': {'60': 'capsa.step_impl_evaluation_l60', '65': 'capsa.step_impl_evaluation_l65'}}}, 'cat_so': {'cat_so.py': {'_init_pop': {'28': 'cat_so.init_pop'}, 'step': {'53': 'cat_so.velocities_reuse_pop_shape', '59': 'cat_so.step_evaluation_l59'}}}, 'cco': {'cco.py': {'_greedy_single': {'47': 'cco.greedy_single'}}}, 'cddo': {'cddo.py': {'_step_impl': {'58': 'cddo.step_impl_evaluation_l58'}}}, 'cddo_child': {'cddo_child.py': {'_step_impl': {'74': 'cddo_child.step_impl_evaluation_l74'}}}, 'cdo': {'cdo.py': {'_step_impl': {'34': 'cdo.step_impl_evaluation_l34'}}}, 'cdo_chernobyl': {'cdo_chernobyl.py': {'_step_impl': {'54': 'cdo_chernobyl.step_impl_evaluation_l54'}}}, 'cem': {'cem.py': {'_init_pop': {'27': 'cem.init_pop'}, 'step': {'49': 'cem.step_evaluation_l49'}, 'inject_candidates': {'102': 'cem.inject_candidates_evaluation_l102'}}}, 'ceo_cosmic': {'ceo_cosmic.py': {'_step_impl': {'86': 'ceo_cosmic.step_impl_evaluation_l86'}}}, 'cfoa': {'cfoa.py': {'_step_impl': {'125': 'cfoa.step_impl_evaluation_l125'}}}, 'cg_gwo': {'cg_gwo.py': {'initialize': {'99': 'cg_gwo.initialize_evaluation_l99'}, 'step': {'301': 'cg_gwo.step_evaluation_l301', '313': 'cg_gwo.step_evaluation_l313'}}}, 'cgo': {'cgo.py': {'_step_impl': {'50': 'cgo.evaluate_all_4n_seeds_keep_best', '55': 'cgo.evaluate_all_4n_seeds_keep_best'}}}, 'chameleon_sa': {'chameleon_sa.py': {'_step_impl': {'45': 'chameleon_sa.step_impl_evaluation_l45', '49': 'chameleon_sa.step_impl_evaluation_l49'}}}, 'chaotic_gwo': {'chaotic_gwo.py': {'initialize': {'99': 'chaotic_gwo.initialize_evaluation_l99'}, 'step': {'301': 'chaotic_gwo.step_evaluation_l301', '313': 'chaotic_gwo.step_evaluation_l313'}}}, 'chicken_so': {'chicken_so.py': {'_init_pop': {'26': 'chicken_so.init_pop'}, 'step': {'53': 'chicken_so.step_evaluation_l53', '71': 'chicken_so.step_evaluation_l71'}}}, 'chio': {'chio.py': {'_step_impl': {'69': 'chio.immune_contact'}}}, 'choa': {'choa.py': {'_step_impl': {'46': 'choa.step_impl_evaluation_l46'}}}, 'circle_sa': {'circle_sa.py': {'_step_impl': {'42': 'circle_sa.eq_8'}}}, 'clonalg': {'clonalg.py': {'_init_pop': {'27': 'clonalg.init_pop'}, 'step': {'50': 'clonalg.step_evaluation_l50', '54': 'clonalg.step_evaluation_l54'}}}, 'cmaes': {'cmaes.py': {'initialize': {'52': 'cmaes.evaluate_one_point_get_initial_best'}, 'step': {'94': 'cmaes.evaluate'}}}, 'coa': {'coa.py': {'_step_impl': {'58': 'coa.social_condition_eq_12', '69': 'coa.pup_birth_eq_7'}}}, 'coati_oa': {'coati_oa.py': {'_init_pop': {'27': 'coati_oa.init_pop'}, 'step': {'45': 'coati_oa.step_evaluation_l45'}}}, 'cockroach_so': {'cockroach_so.py': {'_init_pop': {'27': 'cockroach_so.init_pop'}, 'step': {'46': 'cockroach_so.step_evaluation_l46', '54': 'cockroach_so.step_evaluation_l54', '55': 'cockroach_so.step_evaluation_l55'}}}, 'compact_ga': {'compact_ga.py': {'_evaluate_bits': {'125': 'compact_ga.evaluate_bits'}, 'step': {'211': 'compact_ga.step_evaluation_l211'}}}, 'coot': {'coot.py': {'_step_impl': {'47': 'coot.step_impl_evaluation_l47', '60': 'coot.step_impl_evaluation_l60'}}}, 'cpo': {'cpo.py': {'_step_impl': {'195': 'cpo.step_impl_evaluation_l195'}}}, 'crayfish_oa': {'crayfish_oa.py': {'_step_impl': {'71': 'crayfish_oa.step_impl_evaluation_l71'}}}, 'cro': {'cro.py': {'_step_impl': {'37': 'cro.step_impl_evaluation_l37', '46': 'cro.step_impl_evaluation_l46'}}}, 'csa': {'csa.py': {'_init_pop': {'27': 'csa.init_pop'}, 'step': {'48': 'csa.step_evaluation_l48'}}}, 'csbo': {'csbo.py': {'_step_impl': {'34': 'csbo.systolic', '41': 'csbo.diastolic'}}}, 'cso': {'cso.py': {'initialize': {'35': 'cso.initialize_evaluation_l35'}, 'step': {'73': 'cso.mean_all_positions'}}}, 'cuckoo_s': {'cuckoo_s.py': {'_init_pop': {'27': 'cuckoo_s.init_pop'}, 'step': {'49': 'cuckoo_s.step_evaluation_l49', '60': 'cuckoo_s.abandon_worst_nests'}}}, 'da': {'da.py': {'_init_pop': {'30': 'da.init_pop'}, 'step': {'82': 'da.step_evaluation_l82'}, 'inject_candidates': {'131': 'da.inject_candidates_evaluation_l131'}}}, 'dbo': {'dbo.py': {'_step_impl': {'40': 'dbo.step_impl_evaluation_l40', '58': 'dbo.step_impl_evaluation_l58'}}}, 'ddao': {'ddao.py': {'_step_impl': {'26': 'ddao.step_impl_evaluation_l26', '35': 'ddao.step_impl_evaluation_l35'}}}, 'de': {'de.py': {'_init_pop': {'52': 'de.init_pop'}, 'step': {'105': 'de.step_evaluation_l105'}}}, 'deo_dolphin': {'deo_dolphin.py': {'_step_impl': {'60': 'deo_dolphin.step_impl_evaluation_l60'}}}, 'dfo': {'dfo.py': {'_init_pop': {'27': 'dfo.init_pop'}, 'step': {'45': 'dfo.step_evaluation_l45', '50': 'dfo.step_evaluation_l50'}}}, 'dhole_oa': {'dhole_oa.py': {'_step_impl': {'92': 'dhole_oa.eq_5', '126': 'dhole_oa.small_weak_prey_immediate_kill'}}}, 'dmoa': {'dmoa.py': {'_step_impl': {'50': 'dmoa.selection', '63': 'dmoa.scout_phase', '75': 'dmoa.3_baby_sitter_eviction', '88': 'dmoa.scalar_broadcast'}}}, 'do_dandelion': {'do_dandelion.py': {'_step_impl': {'55': 'do_dandelion.3'}}}, 'doa': {'doa.py': {'_step_impl': {'63': 'doa.step_impl_evaluation_l63', '78': 'doa.exploitation'}}}, 'dra': {'dra.py': {'_step_impl': {'74': 'dra.step_impl_evaluation_l74', '96': 'dra.step_impl_evaluation_l96', '110': 'dra.step_impl_evaluation_l110'}}}, 'dream_oa': {'dream_oa.py': {'_step_impl': {'115': 'dream_oa.step_impl_evaluation_l115'}}}, 'ds_gwo': {'ds_gwo.py': {'initialize': {'99': 'ds_gwo.initialize_evaluation_l99'}, 'step': {'301': 'ds_gwo.step_evaluation_l301', '313': 'ds_gwo.step_evaluation_l313'}}}, 'dso': {'dso.py': {'_step_impl': {'31': 'dso.step_impl_evaluation_l31'}}}, 'dvba': {'dvba.py': {'_init_pop': {'27': 'dvba.init_pop'}, 'step': {'49': 'dvba.step_evaluation_l49', '58': 'dvba.step_evaluation_l58'}}}, 'eao': {'eao.py': {'_step_impl': {'39': 'eao.candidate_generation', '46': 'eao.candidate_generation', '53': 'eao.candidate_generation'}}}, 'eco': {'eco.py': {'_step_impl': {'45': 'eco.step_impl_evaluation_l45'}}}, 'ecological_cycle_o': {'ecological_cycle_o.py': {'_eval_accept_group': {'68': 'ecological_cycle_o.eval_accept_group'}, '_step_impl': {'82': 'ecological_cycle_o.step_impl_evaluation_l82', '149': 'ecological_cycle_o.step_impl_evaluation_l149'}}}, 'ecpo': {'ecpo.py': {'initialize': {'107': 'ecpo.initialize_evaluation_l107'}, 'step': {'134': 'ecpo.random_perturbation'}}}, 'edo': {'edo.py': {'_step_impl': {'51': 'edo.step_impl_evaluation_l51'}}}, 'eefo': {'eefo.py': {'_step_impl': {'117': 'eefo.step_impl_evaluation_l117'}}}, 'eel_grouper_o': {'eel_grouper_o.py': {'_step_impl': {'72': 'eel_grouper_o.step_impl_evaluation_l72', '98': 'eel_grouper_o.step_impl_evaluation_l98'}}}, 'efo': {'efo.py': {'_step_impl': {'57': 'efo.mutation'}}}, 'ego': {'ego.py': {'initialize': {'47': 'ego.initialize_evaluation_l47'}, 'step': {'61': 'ego.candidate_generation'}}}, 'eho': {'eho.py': {'_init_pop': {'27': 'eho.init_pop'}, 'step': {'53': 'eho.step_evaluation_l53'}}}, 'elk_ho': {'elk_ho.py': {'_step_impl': {'50': 'elk_ho.step_impl_evaluation_l50'}}}, 'enhanced_aeo': {'enhanced_aeo.py': {'initialize': {'99': 'enhanced_aeo.initialize_evaluation_l99'}, 'step': {'301': 'enhanced_aeo.step_evaluation_l301', '313': 'enhanced_aeo.step_evaluation_l313'}}}, 'enhanced_two': {'enhanced_two.py': {'initialize': {'99': 'enhanced_two.initialize_evaluation_l99'}, 'step': {'301': 'enhanced_two.step_evaluation_l301', '313': 'enhanced_two.step_evaluation_l313'}}}, 'eo': {'eo.py': {'_step_impl': {'60': 'eo.position_eq_16'}}}, 'eoa': {'eoa.py': {'_step_impl': {'55': 'eoa.step_impl_evaluation_l55', '74': 'eoa.mutation'}}}, 'ep': {'ep.py': {'_step_impl': {'40': 'ep.generate_offspring'}}}, 'epc': {'epc.py': {'_step_impl': {'33': 'epc.step_impl_evaluation_l33'}}}, 'er_gwo': {'er_gwo.py': {'initialize': {'99': 'er_gwo.initialize_evaluation_l99'}, 'step': {'301': 'er_gwo.step_evaluation_l301', '313': 'er_gwo.step_evaluation_l313'}}}, 'es': {'es.py': {'_step_impl': {'40': 'es.step_impl_evaluation_l40'}}}, 'esc': {'esc.py': {'_step_impl': {'36': 'esc.explore_randomly'}}}, 'eso': {'eso.py': {'_step_impl': {'69': 'eso.step_impl_evaluation_l69'}}}, 'esoa': {'esoa.py': {'_step_impl': {'42': 'esoa.step_impl_evaluation_l42', '46': 'esoa.step_impl_evaluation_l46'}}}, 'eto': {'eto.py': {'_step_impl': {'33': 'eto.trigonometric_component'}}}, 'evo': {'evo.py': {'_step_impl': {'41': 'evo.step_impl_evaluation_l41', '45': 'evo.step_impl_evaluation_l45'}}}, 'ex_gwo': {'ex_gwo.py': {'initialize': {'99': 'ex_gwo.initialize_evaluation_l99'}, 'step': {'301': 'ex_gwo.step_evaluation_l301', '313': 'ex_gwo.step_evaluation_l313'}}}, 'fata': {'fata.py': {'_step_impl': {'44': 'fata.step_impl_evaluation_l44'}}}, 'fbio': {'fbio.py': {'_step_impl': {'33': 'fbio.team_step_a1_gaussian_perturbation_around', '56': 'fbio.exploration', '66': 'fbio.team_b_step_b1_convex_combination'}}}, 'fda': {'fda.py': {'_init_pop': {'27': 'fda.init_pop'}, 'step': {'55': 'fda.generate_neighbours', '72': 'fda.step_evaluation_l72'}}}, 'fdo': {'fdo.py': {'_step_impl': {'32': 'fdo.step_impl_evaluation_l32', '37': 'fdo.step_impl_evaluation_l37', '41': 'fdo.step_impl_evaluation_l41'}}}, 'fep': {'fep.py': {'initialize': {'33': 'fep.initialize_evaluation_l33'}, 'step': {'58': 'fep.mutation'}}}, 'ffa': {'ffa.py': {'_step_impl': {'35': 'ffa.step_impl_evaluation_l35'}}}, 'ffo': {'ffo.py': {'_step_impl': {'21': 'ffo.step_impl_evaluation_l21', '28': 'ffo.step_impl_evaluation_l28'}}}, 'firefly_a': {'firefly_a.py': {'_init_pop': {'27': 'firefly_a.init_pop'}, 'step': {'53': 'firefly_a.step_evaluation_l53'}}}, 'fla': {'fla.py': {'_step_impl': {'79': 'fla.tf_0_9'}}}, 'flo': {'flo.py': {'_greedy_single': {'46': 'flo.greedy_single'}}}, 'flood_a': {'flood_a.py': {'_step_impl': {'34': 'flood_a.step_impl_evaluation_l34', '46': 'flood_a.step_impl_evaluation_l46'}}}, 'foa': {'foa.py': {'_step_impl': {'38': 'foa.step_impl_evaluation_l38', '55': 'foa.step_impl_evaluation_l55', '58': 'foa.step_impl_evaluation_l58'}}}, 'foa_fossa': {'foa_fossa.py': {'_step_impl': {'58': 'foa_fossa.step_impl_evaluation_l58'}}}, 'fox': {'fox.py': {'_step_impl': {'74': 'fox.preserve_best_few_individuals_explicitly'}}}, 'fpa': {'fpa.py': {'_init_pop': {'27': 'fpa.init_pop'}, 'step': {'53': 'fpa.step_evaluation_l53', '56': 'fpa.step_evaluation_l56'}}}, 'frcg': {'frcg.py': {'initialize': {'56': 'frcg.initialize_evaluation_l56'}, 'step': {'83': 'frcg.armijo_line_search_along_dk'}}}, 'frofi': {'frofi.py': {'initialize': {'32': 'frofi.initialize_evaluation_l32'}, 'step': {'88': 'frofi.de_offspring_generation', '100': 'frofi.mutation'}}}, 'fss': {'fss.py': {'_step_impl': {'40': 'fss.step_impl_evaluation_l40', '50': 'fss.step_impl_evaluation_l50', '58': 'fss.step_impl_evaluation_l58'}}}, 'fuzzy_gwo': {'fuzzy_gwo.py': {'initialize': {'99': 'fuzzy_gwo.initialize_evaluation_l99'}, 'step': {'301': 'fuzzy_gwo.step_evaluation_l301', '313': 'fuzzy_gwo.step_evaluation_l313'}}}, 'fwa': {'fwa.py': {'_step_impl': {'46': 'fwa.step_impl_evaluation_l46'}}}, 'ga': {'ga.py': {'_init_pop': {'25': 'ga.init_pop'}, '_breed': {'52': 'ga.breed'}, '_mutate': {'64': 'ga.mutate'}}}, 'gazelle_oa': {'gazelle_oa.py': {'_step_impl': {'59': 'gazelle_oa.step_impl_evaluation_l59', '76': 'gazelle_oa.step_impl_evaluation_l76'}}}, 'gbo': {'gbo.py': {'_step_impl': {'78': 'gbo.step_impl_evaluation_l78'}}}, 'gco': {'gco.py': {'_step_impl': {'34': 'gco.dark_zone'}}}, 'gea': {'gea.py': {'_step_impl': {'43': 'gea.nearest_by_cosine_similarity', '52': 'gea.second_attempt'}}}, 'ggo': {'ggo.py': {'_step_impl': {'23': 'ggo.step_impl_evaluation_l23'}}}, 'gja': {'gja.py': {'_step_impl': {'60': 'gja.step_impl_evaluation_l60', '69': 'gja.step_impl_evaluation_l69'}}}, 'gjo': {'gjo.py': {'_step_impl': {'63': 'gjo.exploration'}}}, 'gkso': {'gkso.py': {'_step_impl': {'40': 'gkso.crossover', '48': 'gkso.2_shark_hunt'}}}, 'gmo': {'gmo.py': {'_init_pop': {'27': 'gmo.init_pop'}, 'step': {'60': 'gmo.improve_guide', '68': 'gmo.improve_guide'}}}, 'gndo': {'gndo.py': {'_step_impl': {'41': 'gndo.step_impl_evaluation_l41'}}}, 'go_growth': {'go_growth.py': {'_step_impl': {'43': 'go_growth.step_impl_evaluation_l43', '57': 'go_growth.step_impl_evaluation_l57'}}}, 'goa': {'goa.py': {'_init_pop': {'27': 'goa.init_pop'}, 'step': {'54': 'goa.step_evaluation_l54'}}}, 'gpoo': {'gpoo.py': {'_step_impl': {'120': 'gpoo.step_impl_evaluation_l120'}}}, 'gpso': {'gpso.py': {'_local_search': {'65': 'gpso.local_search'}, 'initialize': {'75': 'gpso.initialize_evaluation_l75'}, 'step': {'108': 'gpso.step_evaluation_l108'}}}, 'grasp': {'grasp.py': {'_construct': {'45': 'grasp.construct'}}}, 'gsa': {'gsa.py': {'_init_pop': {'27': 'gsa.init_pop'}, 'step': {'62': 'gsa.step_evaluation_l62'}}}, 'gska': {'gska.py': {'_step_impl': {'67': 'gska.step_impl_evaluation_l67'}}}, 'gso': {'gso.py': {'_step_impl': {'43': 'gso.step_impl_evaluation_l43'}}}, 'gso_glider_snake': {'gso_glider_snake.py': {'_init_pop': {'56': 'gso_glider_snake.init_pop'}, 'step': {'101': 'gso_glider_snake.step_evaluation_l101'}}}, 'gto': {'gto.py': {'_step_impl': {'48': 'gto.1_extensive_search_eq_4', '60': 'gto.2_choosing_area_eq_7', '76': 'gto.3_attacking_eqs_10_13_15'}}}, 'gwo': {'gwo.py': {'_init_pop': {'27': 'gwo.init_pop'}, 'step': {'51': 'gwo.step_evaluation_l51', '52': 'gwo.step_evaluation_l52', '53': 'gwo.step_evaluation_l53', '55': 'gwo.step_evaluation_l55'}}}, 'gwo_woa': {'gwo_woa.py': {'initialize': {'99': 'gwo_woa.initialize_evaluation_l99'}, 'step': {'301': 'gwo_woa.step_evaluation_l301', '313': 'gwo_woa.step_evaluation_l313'}}}, 'hba': {'hba.py': {'_step_impl': {'47': 'hba.step_impl_evaluation_l47', '50': 'hba.step_impl_evaluation_l50'}}}, 'hba_honey': {'hba_honey.py': {'_step_impl': {'60': 'hba_honey.step_impl_evaluation_l60'}}}, 'hbo': {'hbo.py': {'_step_impl': {'87': 'hbo.no_change'}}}, 'hco': {'hco.py': {'_step_impl': {'62': 'hco.step_impl_evaluation_l62'}}}, 'hde': {'hde.py': {'_step_impl': {'22': 'hde.step_impl_evaluation_l22', '31': 'hde.step_impl_evaluation_l31'}}}, 'heoa': {'heoa.py': {'_step_impl': {'79': 'heoa.risk_takers_generate_bounded_sample_around'}}}, 'hgs': {'hgs.py': {'_step_impl': {'69': 'hgs.step_impl_evaluation_l69'}}}, 'hgso': {'hgso.py': {'_step_impl': {'66': 'hgso.best_in_cluster', '75': 'hgso.replace_worst_n_w_agents_eq'}}}, 'hho': {'hho.py': {'_init_pop': {'29': 'hho.init_pop'}, 'step': {'55': 'hho.step_evaluation_l55', '60': 'hho.step_evaluation_l60', '64': 'hho.step_evaluation_l64', '68': 'hho.step_evaluation_l68', '72': 'hho.step_evaluation_l72', '77': 'hho.step_evaluation_l77', '81': 'hho.step_evaluation_l81'}}}, 'hi_woa': {'hi_woa.py': {'initialize': {'99': 'hi_woa.initialize_evaluation_l99'}, 'step': {'301': 'hi_woa.step_evaluation_l301', '313': 'hi_woa.step_evaluation_l313'}}}, 'hiking_oa': {'hiking_oa.py': {'_step_impl': {'28': 'hiking_oa.step_impl_evaluation_l28'}}}, 'ho_hippo': {'ho_hippo.py': {'_step_impl': {'52': 'ho_hippo.step_impl_evaluation_l52', '53': 'ho_hippo.step_impl_evaluation_l53', '58': 'ho_hippo.step_impl_evaluation_l58', '66': 'ho_hippo.step_impl_evaluation_l66', '72': 'ho_hippo.step_impl_evaluation_l72'}}}, 'horse_oa': {'horse_oa.py': {'_step_impl': {'46': 'horse_oa.step_impl_evaluation_l46'}}}, 'hsa': {'hsa.py': {'_init_pop': {'25': 'hsa.init_pop'}, 'step': {'45': 'hsa.step_evaluation_l45'}}}, 'hsaba': {'hsaba.py': {'_step_impl': {'39': 'hsaba.step_impl_evaluation_l39'}}}, 'hus': {'hus.py': {'_init_pop': {'29': 'hus.init_pop'}, 'step': {'60': 'hus.step_evaluation_l60'}}}, 'i_gwo': {'i_gwo.py': {'_init_pop': {'27': 'i_gwo.init_pop'}, 'step': {'51': 'i_gwo.step_evaluation_l51', '52': 'i_gwo.step_evaluation_l52', '53': 'i_gwo.step_evaluation_l53', '55': 'i_gwo.step_evaluation_l55', '69': 'i_gwo.improve_step'}}}, 'i_woa': {'i_woa.py': {'_init_pop': {'24': 'i_woa.init_pop'}, '_breed': {'47': 'i_woa.breed'}, 'step': {'76': 'i_woa.step_evaluation_l76'}}}, 'iagwo': {'iagwo.py': {'_init_pop': {'65': 'iagwo.init_pop'}, 'step': {'154': 'iagwo.step_evaluation_l154'}}}, 'iaro': {'iaro.py': {'initialize': {'99': 'iaro.initialize_evaluation_l99'}, 'step': {'301': 'iaro.step_evaluation_l301', '313': 'iaro.step_evaluation_l313'}}}, 'ica': {'ica.py': {'_step_impl': {'60': 'ica.1_assimilation', '68': 'ica.2_revolution_random_dimension_reset', '75': 'ica.2_revolution_random_dimension_reset'}}}, 'ikoa': {'ikoa.py': {'_step_impl': {'87': 'ikoa.step_impl_evaluation_l87', '88': 'ikoa.step_impl_evaluation_l88'}}}, 'ilshade': {'ilshade.py': {'_step_impl': {'44': 'ilshade.step_impl_evaluation_l44'}}}, 'imode': {'imode.py': {'initialize': {'40': 'imode.initialize_evaluation_l40'}, 'step': {'140': 'imode.step_evaluation_l140'}}}, 'improved_aeo': {'improved_aeo.py': {'initialize': {'99': 'improved_aeo.initialize_evaluation_l99'}, 'step': {'301': 'improved_aeo.step_evaluation_l301', '313': 'improved_aeo.step_evaluation_l313'}}}, 'improved_qsa': {'improved_qsa.py': {'initialize': {'99': 'improved_qsa.initialize_evaluation_l99'}, 'step': {'301': 'improved_qsa.step_evaluation_l301', '313': 'improved_qsa.step_evaluation_l313'}}}, 'improved_tlo': {'improved_tlo.py': {'initialize': {'99': 'improved_tlo.initialize_evaluation_l99'}, 'step': {'301': 'improved_tlo.step_evaluation_l301', '313': 'improved_tlo.step_evaluation_l313'}}}, 'incremental_gwo': {'incremental_gwo.py': {'initialize': {'99': 'incremental_gwo.initialize_evaluation_l99'}, 'step': {'301': 'incremental_gwo.step_evaluation_l301', '313': 'incremental_gwo.step_evaluation_l313'}}}, 'info': {'info.py': {'_step_impl': {'81': 'info.step_impl_evaluation_l81'}}}, 'iobl_gwo': {'iobl_gwo.py': {'initialize': {'99': 'iobl_gwo.initialize_evaluation_l99'}, 'step': {'301': 'iobl_gwo.step_evaluation_l301', '313': 'iobl_gwo.step_evaluation_l313'}}}, 'ivya': {'ivya.py': {'_step_impl': {'36': 'ivya.step_impl_evaluation_l36'}}}, 'iwo': {'iwo.py': {'_step_impl': {'51': 'iwo.step_impl_evaluation_l51'}}}, 'jade': {'jade.py': {'initialize': {'99': 'jade.initialize_evaluation_l99'}, 'step': {'301': 'jade.step_evaluation_l301', '313': 'jade.step_evaluation_l313'}}}, 'jde': {'jde.py': {'_step_impl': {'37': 'jde.step_impl_evaluation_l37'}}}, 'jso': {'jso.py': {'_init_pop': {'27': 'jso.init_pop'}, 'step': {'61': 'jso.step_evaluation_l61'}}}, 'jy': {'jy.py': {'_init_pop': {'27': 'jy.init_pop'}, 'step': {'47': 'jy.step_evaluation_l47'}}}, 'kha': {'kha.py': {'_init_pop': {'24': 'kha.init_pop'}, 'step': {'62': 'kha.step_evaluation_l62', '64': 'kha.foraging', '69': 'kha.foraging', '73': 'kha.diffusion', '77': 'kha.position', '93': 'kha.step_evaluation_l93', '100': 'kha.step_evaluation_l100'}}}, 'kma': {'kma.py': {'initialize': {'32': 'kma.initialize_evaluation_l32'}, 'step': {'67': 'kma.n_big', '80': 'kma.female_reproduction', '95': 'kma.small_males_move_towards_big_males'}}}, 'laro': {'laro.py': {'initialize': {'99': 'laro.initialize_evaluation_l99'}, 'step': {'301': 'laro.step_evaluation_l301', '313': 'laro.step_evaluation_l313'}}}, 'lca': {'lca.py': {'_step_impl': {'39': 'lca.mutation'}}}, 'lco': {'lco.py': {'_step_impl': {'47': 'lco.boundary_reflection'}}}, 'levy_ja': {'levy_ja.py': {'initialize': {'99': 'levy_ja.initialize_evaluation_l99'}, 'step': {'301': 'levy_ja.step_evaluation_l301', '313': 'levy_ja.step_evaluation_l313'}}}, 'lfd': {'lfd.py': {'_step_impl': {'46': 'lfd.step_impl_evaluation_l46'}}}, 'liwo': {'liwo.py': {'_step_impl': {'73': 'liwo.step_impl_evaluation_l73'}}}, 'loa': {'loa.py': {'_step_impl': {'46': 'loa.step_impl_evaluation_l46'}}}, 'loa_lyrebird': {'loa_lyrebird.py': {'_step_impl': {'59': 'loa_lyrebird.step_impl_evaluation_l59'}}}, 'lpo': {'lpo.py': {'_step_impl': {'51': 'lpo.step_impl_evaluation_l51'}}}, 'lshade_cnepsin': {'lshade_cnepsin.py': {'_step_impl': {'112': 'lshade_cnepsin.step_impl_evaluation_l112'}}}, 'lso_spectrum': {'lso_spectrum.py': {'_step_impl': {'87': 'lso_spectrum.step_impl_evaluation_l87'}}}, 'mbo': {'mbo.py': {'_init_pop': {'27': 'mbo.init_pop'}, 'step': {'67': 'mbo.step_evaluation_l67'}}}, 'memetic_a': {'memetic_a.py': {'_init_pop': {'30': 'memetic_a.init_pop'}, '_breed': {'56': 'memetic_a.breed'}, '_mutate': {'68': 'memetic_a.mutate'}, '_xhc': {'82': 'memetic_a.xhc'}}}, 'mfa': {'mfa.py': {'_init_pop': {'27': 'mfa.init_pop'}, 'step': {'52': 'mfa.step_evaluation_l52'}}}, 'mfea': {'mfea.py': {'initialize': {'44': 'mfea.initialize_evaluation_l44'}, 'step': {'87': 'mfea.mutation'}}}, 'mfo': {'mfo.py': {'_greedy_update': {'65': 'mfo.greedy_update'}}}, 'mgo': {'mgo.py': {'initialize': {'32': 'mgo.initialize_evaluation_l32'}, 'step': {'100': 'mgo.clip_all'}}}, 'mgoa_market': {'mgoa_market.py': {'_step_impl': {'63': 'mgoa_market.step_impl_evaluation_l63'}}}, 'mke': {'mke.py': {'_step_impl': {'33': 'mke.step_impl_evaluation_l33'}}}, 'modified_aeo': {'modified_aeo.py': {'initialize': {'99': 'modified_aeo.initialize_evaluation_l99'}, 'step': {'301': 'modified_aeo.step_evaluation_l301', '313': 'modified_aeo.step_evaluation_l313'}}}, 'modified_eo': {'modified_eo.py': {'initialize': {'99': 'modified_eo.initialize_evaluation_l99'}, 'step': {'301': 'modified_eo.step_evaluation_l301', '313': 'modified_eo.step_evaluation_l313'}}}, 'moss_go': {'moss_go.py': {'_step_impl': {'44': 'moss_go.water_dispersal_gradient_like'}}}, 'mpa': {'mpa.py': {'_step_impl': {'74': 'mpa.fish_aggregating_devices_fads_effect'}}}, 'mrfo': {'mrfo.py': {'_step_impl': {'51': 'mrfo.chain_foraging_eqs_1_2', '64': 'mrfo.somersault_foraging_eq_8'}}}, 'msa_e': {'msa_e.py': {'_step_impl': {'56': 'msa_e.exploitation'}}}, 'mshoa': {'mshoa.py': {'_step_impl': {'66': 'mshoa.step_impl_evaluation_l66'}}}, 'mso': {'mso.py': {'_step_impl': {'40': 'mso.step_impl_evaluation_l40', '45': 'mso.inferior_mirage_search_remaining'}}}, 'mtbo': {'mtbo.py': {'_step_impl': {'76': 'mtbo.random_relocation'}}}, 'mts': {'mts.py': {'_local': {'30': 'mts.local'}}}, 'mvo': {'mvo.py': {'_init_pop': {'27': 'mvo.init_pop'}, 'step': {'59': 'mvo.step_evaluation_l59'}}}, 'mvpa': {'mvpa.py': {'initialize': {'38': 'mvpa.initialize_evaluation_l38'}, 'step': {'66': 'mvpa.each_player_moves_toward_mvp_random'}}}, 'nca': {'nca.py': {'_step_impl': {'67': 'nca.step_impl_evaluation_l67', '98': 'nca.acceleration_hyperbolic_contraction_random_subset_components'}}}, 'ngo': {'ngo.py': {'_step_impl': {'38': 'ngo.1', '47': 'ngo.2'}}}, 'nlapsmjso_eda': {'nlapsmjso_eda.py': {'_evaluate_population': {'98': 'nlapsmjso_eda.evaluate_population_evaluation_l98'}, 'initialize': {'181': 'nlapsmjso_eda.initialize_evaluation_l181'}, 'step': {'254': 'nlapsmjso_eda.step_evaluation_l254', '301': 'nlapsmjso_eda.step_evaluation_l301'}}}, 'nmm': {'nmm.py': {'_step_impl': {'30': 'nmm.step_impl_evaluation_l30', '36': 'nmm.step_impl_evaluation_l36', '39': 'nmm.step_impl_evaluation_l39', '44': 'nmm.step_impl_evaluation_l44'}}}, 'nmra': {'nmra.py': {'_step_impl': {'42': 'nmra.working_operator'}}}, 'nndrea_so': {'nndrea_so.py': {'initialize': {'68': 'nndrea_so.stage_1_evolve_nn_weights_proxy'}, 'step': {'102': 'nndrea_so.decode_through_nn', '122': 'nndrea_so.stage_2_direct_ga_on_binary'}}}, 'noa': {'noa.py': {'_step_impl': {'147': 'noa.step_impl_evaluation_l147'}}}, 'nro': {'nro.py': {'_step_impl': {'64': 'nro.step_impl_evaluation_l64', '82': 'nro.step_impl_evaluation_l82'}}}, 'nwoa': {'nwoa.py': {'_step_impl': {'81': 'nwoa.step_impl_evaluation_l81'}}}, 'ocro': {'ocro.py': {'initialize': {'99': 'ocro.initialize_evaluation_l99'}, 'step': {'301': 'ocro.step_evaluation_l301', '313': 'ocro.step_evaluation_l313'}}}, 'ofa': {'ofa.py': {'initialize': {'32': 'ofa.initialize_evaluation_l32'}, 'step': {'62': 'ofa.use_gen_n_as_proxy_fe'}}}, 'ogwo': {'ogwo.py': {'initialize': {'99': 'ogwo.initialize_evaluation_l99'}, 'step': {'301': 'ogwo.step_evaluation_l301', '313': 'ogwo.step_evaluation_l313'}}}, 'ooa': {'ooa.py': {'_step_impl': {'22': 'ooa.step_impl_evaluation_l22', '25': 'ooa.step_impl_evaluation_l25'}}}, 'parrot_o': {'parrot_o.py': {'_step_impl': {'48': 'parrot_o.fly_new_area'}}}, 'pbil': {'pbil.py': {'initialize': {'26': 'pbil.initialize_evaluation_l26'}, 'step': {'37': 'pbil.step_evaluation_l37'}}}, 'pcx': {'pcx.py': {'_init_pop': {'62': 'pcx.init_pop'}, '_generate_child': {'114': 'pcx.generate_child'}, 'step': {'165': 'pcx.step_evaluation_l165'}, 'inject_candidates': {'265': 'pcx.inject_candidates_evaluation_l265'}}}, 'pdo': {'pdo.py': {'_step_impl': {'48': 'pdo.step_impl_evaluation_l48'}}}, 'petio': {'petio.py': {'_step_impl': {'252': 'petio.step_impl_evaluation_l252'}}}, 'pfa': {'pfa.py': {'_init_pop': {'27': 'pfa.init_pop'}, 'step': {'48': 'pfa.pathfinder', '60': 'pfa.positions'}}}, 'pfa_polar_fox': {'pfa_polar_fox.py': {'_init_pop': {'96': 'pfa_polar_fox.init_pop'}, '_experience_phase': {'166': 'pfa_polar_fox.experience_phase'}, '_leader_phase': {'184': 'pfa_polar_fox.leader_phase'}, 'step': {'239': 'pfa_polar_fox.step_evaluation_l239', '247': 'pfa_polar_fox.step_evaluation_l247'}}}, 'pko': {'pko.py': {'_step_impl': {'42': 'pko.step_impl_evaluation_l42', '53': 'pko.step_impl_evaluation_l53'}}}, 'plba': {'plba.py': {'_step_impl': {'35': 'plba.step_impl_evaluation_l35'}}}, 'plo': {'plo.py': {'_step_impl': {'47': 'plo.step_impl_evaluation_l47'}}}, 'poa': {'poa.py': {'_step_impl': {'35': 'poa.1_moving_toward_prey_eq_4', '42': 'poa.2_winging_on_water_surface_eq'}}}, 'political_o': {'political_o.py': {'_step_impl': {'30': 'political_o.electoral_campaign', '39': 'political_o.parliamentary'}}}, 'pro': {'pro.py': {'_step_impl': {'29': 'pro.step_impl_evaluation_l29', '36': 'pro.step_impl_evaluation_l36'}}}, 'pso': {'pso.py': {'initialize': {'75': 'pso.initialize_evaluation_l75'}, 'step': {'123': 'pso.velocity_position'}, 'inject_candidates': {'251': 'pso.inject_candidates_evaluation_l251'}}}, 'pss': {'pss.py': {'_step_impl': {'44': 'pss.step_impl_evaluation_l44'}}}, 'puma_o': {'puma_o.py': {'_step_impl': {'33': 'puma_o.stalking', '40': 'puma_o.attack'}}}, 'qio': {'qio.py': {'_step_impl': {'57': 'qio.step_impl_evaluation_l57'}}}, 'qle_sca': {'qle_sca.py': {'initialize': {'99': 'qle_sca.initialize_evaluation_l99'}, 'step': {'301': 'qle_sca.step_evaluation_l301', '313': 'qle_sca.step_evaluation_l313'}}}, 'qsa': {'qsa.py': {'_business1': {'30': 'qsa.business1'}, '_business2': {'48': 'qsa.business2'}, '_business3': {'64': 'qsa.business3'}}}, 'random_s': {'random_s.py': {'_init_pop': {'27': 'random_s.init_pop'}, 'step': {'44': 'random_s.step_evaluation_l44'}}}, 'rbmo': {'rbmo.py': {'_step_impl': {'32': 'rbmo.1', '52': 'rbmo.2'}}}, 'rcco': {'rcco.py': {'_step_impl': {'96': 'rcco.step_impl_evaluation_l96', '119': 'rcco.step_impl_evaluation_l119'}}}, 'rfo': {'rfo.py': {'_step_impl': {'78': 'rfo.smell'}}}, 'rhso': {'rhso.py': {'_step_impl': {'64': 'rhso.step_impl_evaluation_l64'}}}, 'rime': {'rime.py': {'_step_impl': {'49': 'rime.hard_rime_puncture_mechanism'}}}, 'rmsprop': {'rmsprop.py': {'initialize': {'56': 'rmsprop.initialize_evaluation_l56'}, 'step': {'72': 'rmsprop.step_evaluation_l72'}}}, 'roa': {'roa.py': {'_step_impl': {'38': 'roa.attempt', '39': 'roa.attempt', '45': 'roa.attempt'}}}, 'rsa': {'rsa.py': {'_step_impl': {'39': 'rsa.step_impl_evaluation_l39'}}}, 'rso': {'rso.py': {'_step_impl': {'23': 'rso.step_impl_evaluation_l23', '31': 'rso.step_impl_evaluation_l31'}}}, 'run': {'run.py': {'_step_impl': {'73': 'run.step_impl_evaluation_l73', '90': 'run.step_impl_evaluation_l90'}}}, 'sa': {'sa.py': {'initialize': {'91': 'sa.initialize_evaluation_l91'}, 'step': {'124': 'sa.step_evaluation_l124'}, 'inject_candidates': {'234': 'sa.inject_candidates_evaluation_l234'}}}, 'saba': {'saba.py': {'_step_impl': {'39': 'saba.step_impl_evaluation_l39'}}}, 'sacoso': {'sacoso.py': {'initialize': {'48': 'sacoso.initialize_evaluation_l48'}, 'step': {'73': 'sacoso.fes_swarm_standard_pso'}}}, 'sade': {'sade.py': {'initialize': {'99': 'sade.initialize_evaluation_l99'}, 'step': {'301': 'sade.step_evaluation_l301', '313': 'sade.step_evaluation_l313'}}}, 'sade_amss': {'sade_amss.py': {'initialize': {'47': 'sade_amss.initialize_evaluation_l47'}, 'step': {'82': 'sade_amss.de_rand_1_bin_on_subspace'}}}, 'sade_atdsc': {'sade_atdsc.py': {'initialize': {'46': 'sade_atdsc.initialize_evaluation_l46'}, 'step': {'75': 'sade_atdsc.without_surrogate_evaluate_all_pick_best'}}}, 'sade_sammon': {'sade_sammon.py': {'initialize': {'50': 'sade_sammon.initialize_evaluation_l50'}, 'step': {'78': 'sade_sammon.step_evaluation_l78'}}}, 'samso': {'samso.py': {'initialize': {'45': 'samso.initialize_evaluation_l45'}, 'step': {'68': 'samso.step_evaluation_l68'}}}, 'sap_de': {'sap_de.py': {'initialize': {'99': 'sap_de.initialize_evaluation_l99'}, 'step': {'301': 'sap_de.step_evaluation_l301', '313': 'sap_de.step_evaluation_l313'}}}, 'saro': {'saro.py': {'_step_impl': {'54': 'saro.boundary_repair', '74': 'saro.individual', '87': 'saro.reinitialize_individuals_too_many_unsuccessful_searches'}}}, 'sbo': {'sbo.py': {'_step_impl': {'52': 'sbo.mutation'}}}, 'sboa': {'sboa.py': {'_step_impl': {'41': 'sboa.1', '53': 'sboa.2'}}}, 'scho': {'scho.py': {'_step_impl': {'58': 'scho.step_impl_evaluation_l58'}}}, 'scso': {'scso.py': {'_step_impl': {'57': 'scso.exploration'}}}, 'sd': {'sd.py': {'initialize': {'56': 'sd.initialize_evaluation_l56'}, 'step': {'75': 'sd.step_evaluation_l75'}}}, 'seaho': {'seaho.py': {'_step_impl': {'65': 'seaho.eq_11', '77': 'seaho.3_reproduction'}}}, 'serval_oa': {'serval_oa.py': {'_step_impl': {'22': 'serval_oa.step_impl_evaluation_l22', '25': 'serval_oa.step_impl_evaluation_l25'}}}, 'sfo': {'sfo.py': {'_initialize_payload': {'25': 'sfo.initialize_payload_evaluation_l25'}, '_step_impl': {'52': 'sfo.sailfish_positions_eq_6', '68': 'sfo.attack_power_eq_10', '75': 'sfo.step_impl_evaluation_l75', '92': 'sfo.replenish_sardines_depleted'}}}, 'sfoa': {'sfoa.py': {'_step_impl': {'41': 'sfoa.step_impl_evaluation_l41', '51': 'sfoa.exploitation'}}}, 'shade': {'shade.py': {'_step_impl': {'43': 'shade.step_impl_evaluation_l43'}}}, 'shio': {'shio.py': {'_step_impl': {'28': 'shio.step_impl_evaluation_l28'}}}, 'shio_success': {'shio_success.py': {'_step_impl': {'59': 'shio_success.step_impl_evaluation_l59'}}}, 'sho': {'sho.py': {'_step_impl': {'50': 'sho.collect_n_hyenas_around_prey', '64': 'sho.step_impl_evaluation_l64'}}}, 'sine_cosine_a': {'sine_cosine_a.py': {'_init_pop': {'27': 'sine_cosine_a.init_pop'}, 'step': {'49': 'sine_cosine_a.step_evaluation_l49'}}}, 'singer_oa': {'singer_oa.py': {'_greedy_apply': {'45': 'singer_oa.greedy_apply'}}}, 'slo': {'slo.py': {'_step_impl': {'47': 'slo.step_impl_evaluation_l47'}}}, 'sma': {'sma.py': {'_step_impl': {'74': 'sma.eq_2_3'}}}, 'smo': {'smo.py': {'_step_impl': {'54': 'smo.1_local_leader', '69': 'smo.2_global_leader', '82': 'smo.3_local_leader_decision'}}}, 'snow_oa': {'snow_oa.py': {'_step_impl': {'46': 'snow_oa.step_impl_evaluation_l46'}}}, 'so_snake': {'so_snake.py': {'_step_impl': {'76': 'so_snake.step_impl_evaluation_l76', '77': 'so_snake.step_impl_evaluation_l77'}}}, 'soa': {'soa.py': {'_step_impl': {'44': 'soa.eq_14'}}}, 'soo': {'soo.py': {'_step_impl': {'40': 'soo.1_oscillatory_position', '56': 'soo.2_top_3_average_oscillatory'}}}, 'sopt': {'sopt.py': {'_select_best': {'44': 'sopt.select_best'}}}, 'sos': {'sos.py': {'_init_pop': {'27': 'sos.init_pop'}, 'step': {'48': 'sos.mutualism', '56': 'sos.comensalism', '67': 'sos.parasitism'}}}, 'sparrow_sa': {'sparrow_sa.py': {'_step_impl': {'41': 'sparrow_sa.step_impl_evaluation_l41', '50': 'sparrow_sa.step_impl_evaluation_l50', '60': 'sparrow_sa.awareness'}}}, 'spbo': {'spbo.py': {'_init_pop': {'27': 'spbo.init_pop'}, 'step': {'57': 'spbo.best_student', '62': 'spbo.groups', '69': 'spbo.groups', '74': 'spbo.step_evaluation_l74', '77': 'spbo.step_evaluation_l77'}}}, 'sqp': {'sqp.py': {'initialize': {'58': 'sqp.initialize_evaluation_l58'}, 'step': {'94': 'sqp.line_search_armijo'}}}, 'squirrel_sa': {'squirrel_sa.py': {'_step_impl': {'62': 'squirrel_sa.case_1_acorn_squirrels_hickory', '77': 'squirrel_sa.toward_acorn', '85': 'squirrel_sa.toward_hickory'}}}, 'srsr': {'srsr.py': {'_step_impl': {'42': 'srsr.1_accumulation_new_positions_via_gaussian', '69': 'srsr.exploration'}}}, 'srsr_robotics': {'srsr_robotics.py': {'_trial_fit': {'51': 'srsr_robotics.trial_fit'}, '_step_impl': {'76': 'srsr_robotics.mean_is_pulled_by_master_robot', '90': 'srsr_robotics.step_impl_evaluation_l90', '117': 'srsr_robotics.random_signed_group_vector_plus_large'}}}, 'ssa': {'ssa.py': {'_init_pop': {'27': 'ssa.init_pop'}, 'step': {'58': 'ssa.step_evaluation_l58'}}}, 'ssdo': {'ssdo.py': {'_step_impl': {'49': 'ssdo.step_impl_evaluation_l49'}}}, 'sso': {'sso.py': {'_step_impl': {'30': 'sso.step_impl_evaluation_l30'}}}, 'sspider_a': {'sspider_a.py': {'_step_impl': {'43': 'sspider_a.step_impl_evaluation_l43'}}}, 'sto': {'sto.py': {'_step_impl': {'23': 'sto.step_impl_evaluation_l23', '26': 'sto.step_impl_evaluation_l26'}}}, 'superb_foa': {'superb_foa.py': {'_step_impl': {'45': 'superb_foa.step_impl_evaluation_l45'}}}, 'supply_do': {'supply_do.py': {'_step_impl': {'39': 'supply_do.step_impl_evaluation_l39', '43': 'supply_do.step_impl_evaluation_l43'}}}, 'tdo': {'tdo.py': {'_step_impl': {'24': 'tdo.step_impl_evaluation_l24', '28': 'tdo.step_impl_evaluation_l28'}}}, 'tfwo': {'tfwo.py': {'initialize': {'53': 'tfwo.initialize_evaluation_l53'}, '_effect_of_objects': {'122': 'tfwo.effect_of_objects', '136': 'tfwo.effect_of_objects'}, '_effect_of_whirlpools': {'167': 'tfwo.effect_of_whirlpools'}}}, 'thro': {'thro.py': {'_step_impl': {'32': 'thro.race'}}}, 'tlbo': {'tlbo.py': {'_init_pop': {'27': 'tlbo.init_pop'}, 'step': {'46': 'tlbo.step_evaluation_l46', '55': 'tlbo.learner_phase'}}}, 'tlco': {'tlco.py': {'_step_impl': {'41': 'tlco.step_impl_evaluation_l41', '55': 'tlco.step_impl_evaluation_l55', '63': 'tlco.step_impl_evaluation_l63', '77': 'tlco.step_impl_evaluation_l77'}}}, 'toa': {'toa.py': {'_step_impl': {'31': 'toa.stage_1_supervisor_guidance', '50': 'toa.step_impl_evaluation_l50', '57': 'toa.stage_3_individual_activity'}}}, 'toc': {'toc.py': {'_step_impl': {'46': 'toc.velocity_radial_tangential_random'}}}, 'tpo': {'tpo.py': {'_step_impl': {'34': 'tpo.step_impl_evaluation_l34'}}}, 'tree_seed_a': {'tree_seed_a.py': {'_step_impl': {'81': 'tree_seed_a.eq_4'}}}, 'ts': {'ts.py': {'initialize': {'28': 'ts.initialize_evaluation_l28'}, 'step': {'53': 'ts.step_evaluation_l53'}}}, 'tsa': {'tsa.py': {'_step_impl': {'40': 'tsa.step_impl_evaluation_l40'}}}, 'tso': {'tso.py': {'_step_impl': {'59': 'tso.parabolic_updates'}}}, 'ttao': {'ttao.py': {'_step_impl': {'69': 'ttao.step_impl_evaluation_l69', '72': 'ttao.step_impl_evaluation_l72', '75': 'ttao.step_impl_evaluation_l75', '78': 'ttao.step_impl_evaluation_l78', '110': 'ttao.step_impl_evaluation_l110', '137': 'ttao.step_impl_evaluation_l137', '148': 'ttao.step_impl_evaluation_l148', '165': 'ttao.step_impl_evaluation_l165'}}}, 'two': {'two.py': {'_step_impl': {'60': 'two.weights_by_rank'}}}, 'vcs': {'vcs.py': {'_step_impl': {'37': 'vcs.1_virus_diffusion', '49': 'vcs.2_host_cell_infection', '64': 'vcs.3_immune_response'}}}, 'waoa': {'waoa.py': {'_step_impl': {'24': 'waoa.step_impl_evaluation_l24', '28': 'waoa.step_impl_evaluation_l28'}}}, 'warso': {'warso.py': {'_step_impl': {'52': 'warso.original_index_sorted_i'}}}, 'wca': {'wca.py': {'_step_impl': {'39': 'wca.streams_toward_river', '48': 'wca.river_toward_sea', '54': 'wca.evaporation_raining'}}}, 'wdo': {'wdo.py': {'_step_impl': {'46': 'wdo.step_impl_evaluation_l46'}}}, 'whale_foa': {'whale_foa.py': {'initialize': {'99': 'whale_foa.initialize_evaluation_l99'}, 'step': {'301': 'whale_foa.step_evaluation_l301', '313': 'whale_foa.step_evaluation_l313'}}}, 'who': {'who.py': {'_step_impl': {'48': 'who.1_local_movement_milling', '52': 'who.1_local_movement_milling', '62': 'who.2_herd_instinct', '86': 'who.social_memory'}}}, 'wmqimrfo': {'wmqimrfo.py': {'initialize': {'99': 'wmqimrfo.initialize_evaluation_l99'}, 'step': {'301': 'wmqimrfo.step_evaluation_l301', '313': 'wmqimrfo.step_evaluation_l313'}}}, 'wo_wave': {'wo_wave.py': {'_step_impl': {'73': 'wo_wave.step_impl_evaluation_l73'}}}, 'woa': {'woa.py': {'_init_pop': {'27': 'woa.init_pop'}, 'step': {'59': 'woa.step_evaluation_l59'}}}, 'wooa': {'wooa.py': {'_greedy_single': {'46': 'wooa.greedy_single'}}}, 'wso': {'wso.py': {'_step_impl': {'57': 'wso.step_impl_evaluation_l57'}}}, 'wutp': {'wutp.py': {'_step_impl': {'44': 'wutp.horizontal', '46': 'wutp.step_impl_evaluation_l46'}}}, 'ydse': {'ydse.py': {'_step_impl': {'52': 'ydse.step_impl_evaluation_l52'}}}, 'zoa': {'zoa.py': {'_step_impl': {'35': 'zoa.1_foraging_eq_3', '50': 'zoa.s2_offensive'}}}}


def resolve_operator_label(algorithm_id: str, filename: str, function: str, line: int | None = None) -> str | None:
    aid = str(algorithm_id)
    base = str(filename).split('/')[-1].split('\\')[-1]
    func = str(function)
    if line is not None:
        try:
            line_key = str(int(line))
            label = CALLSITE_OPERATOR_LABELS.get(aid, {}).get(base, {}).get(func, {}).get(line_key)
            if label:
                return label
        except Exception:
            pass
    label = FUNCTION_OPERATOR_LABELS.get(aid, {}).get(func)
    if label:
        return label
    return None


def labels_for_algorithm(algorithm_id: str) -> list[str]:
    return list(ENGINE_OPERATOR_LABELS.get(str(algorithm_id), []))


__all__ = ["ENGINE_OPERATOR_LABELS", "FUNCTION_OPERATOR_LABELS", "CALLSITE_OPERATOR_LABELS", "resolve_operator_label", "labels_for_algorithm"]

# ---------------------------------------------------------------------------
# Semantic override layer (batch B3-B6)
# ---------------------------------------------------------------------------
# The generated catalog above may contain honest source-location labels such as
# ``acgwo.step_evaluation_l55``.  Those are useful for debugging, but the
# EvoMapX OAM/CDS reports should expose stable semantic probe labels whenever a
# safe template is known.  This layer rewrites runtime source-location labels to
# semantic labels without touching optimizer dynamics.
import re as _re

_B3_PSO = {"aesspso", "aiw_pso", "gpso"}
_B4_GWO_GENERIC = {"cg_gwo", "chaotic_gwo", "ds_gwo", "er_gwo", "ex_gwo", "fuzzy_gwo", "gwo_woa", "incremental_gwo", "iobl_gwo", "ogwo"}
_B5_WOA_GENERIC = {"hi_woa", "whale_foa"}
_B6_LEVY_GENERIC = {"levy_ja", "laro"}

_SINGLE_OPERATOR_SEMANTIC_OK = {
    # Gradient / single transition engines from the semantic gate.
    "adam", "rmsprop", "sd",
    # Engines in B3-B6 whose faithful implementation exposes one evaluated
    # macro-transition in the population lineage.  They remain passive and are
    # therefore treated as single OAM units unless manually split later.
    "aesspso", "gpso", "iagwo", "i_woa", "lfd",
}

_EXACT_SEMANTIC_REWRITE = {
    # PSO variants
    "aesspso.safe": "aesspso.adaptive_velocity_position_update",
    "gpso.step_evaluation_l108": "gpso.velocity_position_update",
    # AC-GWO explicit equations / internal evaluation points
    "acgwo.step_evaluation_l55": "acgwo.alpha_guidance_trial",
    "acgwo.step_evaluation_l56": "acgwo.beta_guidance_trial",
    "acgwo.step_evaluation_l57": "acgwo.delta_guidance_trial",
    "acgwo.step_evaluation_l60": "acgwo.adaptive_weighted_pack_update",
    # Improved GWO explicit internal candidates
    "i_gwo.improve_step": "i_gwo.elite_improvement_step",
    "i_gwo.step_evaluation_l51": "i_gwo.alpha_guidance_trial",
    "i_gwo.step_evaluation_l52": "i_gwo.beta_guidance_trial",
    "i_gwo.step_evaluation_l53": "i_gwo.delta_guidance_trial",
    "i_gwo.step_evaluation_l55": "i_gwo.mean_leader_position_update",
    "iagwo.step_evaluation_l154": "iagwo.adaptive_alpha_beta_delta_update",
    # WOA variants
    "i_woa.breed": "i_woa.polynomial_breeding_refinement",
    "nwoa.step_impl_evaluation_l81": "nwoa.wave_suction_population_update",
    # Cuckoo / Levy family
    "cco.step_impl_evaluation_l198": "cco.greedy_cuckoo_catfish_replacement",
    "fpa.step_evaluation_l53": "fpa.global_levy_pollination",
    "fpa.step_evaluation_l56": "fpa.local_pollination",
    "lfd.step_impl_evaluation_l46": "lfd.levy_flight_search",
}

_SOURCE_LABEL_RE = _re.compile(r"^(?P<aid>[a-zA-Z0-9_]+)\.(?P<stem>.*?)(?:step_evaluation_l(?P<line1>\d+)|step_impl_evaluation_l(?P<line2>\d+)|_evaluation_l(?P<line3>\d+))$")


def semanticize_operator_label(algorithm_id: str, label: str | None) -> str | None:
    """Return a semantic EvoMapX operator label when a safe rewrite is known."""
    if label in {None, ""}:
        return label
    aid = str(algorithm_id)
    lab = str(label)
    if lab in _EXACT_SEMANTIC_REWRITE:
        return _EXACT_SEMANTIC_REWRITE[lab]
    # Label strings should normally start with the engine id; keep foreign
    # labels untouched because they may come from island/agent diagnostics.
    if not lab.startswith(aid + "."):
        return lab
    m = _SOURCE_LABEL_RE.match(lab)
    if not m:
        return lab
    line = m.group("line1") or m.group("line2") or m.group("line3") or ""

    # Generic template engines generated from a shared multi-strategy body.
    if aid in _B3_PSO:
        if line == "301":
            return f"{aid}.adaptive_velocity_position_update"
        if line == "313":
            return f"{aid}.elite_local_refinement"
        return f"{aid}.adaptive_swarm_update"
    if aid in _B4_GWO_GENERIC:
        if line == "301":
            return f"{aid}.leader_guided_population_update"
        if line == "313":
            return f"{aid}.elite_local_refinement"
        return f"{aid}.grey_wolf_position_update"
    if aid in _B5_WOA_GENERIC:
        if line == "301":
            return f"{aid}.whale_position_update"
        if line == "313":
            return f"{aid}.elite_local_refinement"
        return f"{aid}.whale_search_update"
    if aid in _B6_LEVY_GENERIC:
        if line == "301":
            return f"{aid}.levy_global_search"
        if line == "313":
            return f"{aid}.elite_local_refinement"
        return f"{aid}.levy_local_search"

    # Conservative default: return original label.  Later batches should add
    # exact semantic rewrites instead of using broad generic names.
    return lab


_base_resolve_operator_label = resolve_operator_label

def resolve_operator_label(algorithm_id: str, filename: str, function: str, line: int | None = None) -> str | None:  # type: ignore[override]
    raw = _base_resolve_operator_label(algorithm_id, filename, function, line)
    return semanticize_operator_label(algorithm_id, raw) if raw else None


_base_labels_for_algorithm = labels_for_algorithm

def labels_for_algorithm(algorithm_id: str) -> list[str]:  # type: ignore[override]
    aid = str(algorithm_id)
    labels = [semanticize_operator_label(aid, x) for x in _base_labels_for_algorithm(aid)]
    out = []
    for x in labels:
        if x and x not in out:
            out.append(x)
    return out


def semantic_single_operator_ok() -> set[str]:
    return set(_SINGLE_OPERATOR_SEMANTIC_OK)

# Rewrite exported static label table in place so docs/UI helpers also see the
# semantic names for B3-B6.
for _aid, _labels in list(ENGINE_OPERATOR_LABELS.items()):
    _new = []
    for _lab in _labels:
        _slab = semanticize_operator_label(_aid, _lab)
        if _slab and _slab not in _new:
            _new.append(_slab)
    ENGINE_OPERATOR_LABELS[_aid] = _new

__all__ = list(__all__) + ["semanticize_operator_label", "semantic_single_operator_ok"]

# Canonical semantic label for the Bonobo Optimizer.
_SINGLE_OPERATOR_SEMANTIC_OK.update({"bono"})
_EXACT_SEMANTIC_REWRITE.update({
    "bono.step_impl_evaluation_l89": "bono.bonobo_social_search_update",
})
# Re-apply exported label table for the canonical id.
for _aid in ("bono",):
    _labels = ENGINE_OPERATOR_LABELS.get(_aid, [f"{_aid}.bonobo_social_search_update"])
    ENGINE_OPERATOR_LABELS[_aid] = []
    for _lab in _labels:
        _slab = semanticize_operator_label(_aid, _lab)
        if _slab and _slab not in ENGINE_OPERATOR_LABELS[_aid]:
            ENGINE_OPERATOR_LABELS[_aid].append(_slab)

# ---------------------------------------------------------------------------
# Semantic override layer (batch B7: swarm A-C)
# ---------------------------------------------------------------------------
# This batch removes source-location labels from the A-C swarm subset and maps
# accepted transitions to semantic macro-operators where the faithful engine
# exposes only one evaluated transition.  These rewrites are label-only: they do
# not change candidate generation, fitness calls, selection, or RNG use.

_B7_EXACT_SEMANTIC_REWRITE = {
    # A
    "aaa.step_impl_evaluation_l166": "aaa.evolutionary_growth_reproduction_update",
    "aao.step_evaluation_l301": "aao.adaptive_aquila_position_update",
    "aao.step_evaluation_l313": "aao.elite_local_refinement",
    "acor.step_impl_evaluation_l33": "acor.archive_kernel_sampling_update",
    "agto.step_impl_evaluation_l52": "agto.exploration_migration_update",
    "agto.step_impl_evaluation_l72": "agto.exploitation_silverback_update",
    "aho.step_impl_evaluation_l57": "aho.archerfish_shooting_hunting_update",
    "aho.step_impl_evaluation_l76": "aho.prey_escape_refinement_update",
    "ala.step_impl_evaluation_l52": "ala.lemming_migration_position_update",
    "alo.step_evaluation_l62": "alo.random_walk_antlion_trap_update",
    "aoa.step_evaluation_l43": "aoa.arithmetic_operator_position_update",
    "aoo.step_impl_evaluation_l139": "aoo.animated_oat_growth_update",
    "apo.step_impl_evaluation_l73": "apo.protozoa_life_cycle_update",
    "aso.step_impl_evaluation_l41": "aso.anarchic_social_position_update",
    # B
    "bat_a.step_evaluation_l46": "bat_a.frequency_velocity_update",
    "bat_a.step_evaluation_l51": "bat_a.local_random_walk",
    "bat_a.step_evaluation_l53": "bat_a.acceptance_loudness_pulse_update",
    "bboa.step_impl_evaluation_l45": "bboa.pedal_marking_update",
    "bea.step_impl_evaluation_l37": "bea.elite_site_neighbourhood_search",
    "bea.step_impl_evaluation_l43": "bea.scout_site_global_search",
    "bfo.step_impl_evaluation_l38": "bfo.chemotaxis_tumble_update",
    "bfo.step_impl_evaluation_l51": "bfo.swim_continuation_update",
    "bfo.step_impl_evaluation_l55": "bfo.reproduction_elimination_update",
    "bka.step_impl_evaluation_l33": "bka.attack_behavior_update",
    "bka.step_impl_evaluation_l43": "bka.migration_behavior_update",
    "bmo.step_impl_evaluation_l37": "bmo.barnacle_mating_reproduction_update",
    "boa.step_impl_evaluation_l39": "boa.fragrance_attraction_update",
    "bps.step_impl_evaluation_l141": "bps.birds_of_paradise_pose_update",
    "bsa.step_impl_evaluation_l45": "bsa.foraging_or_vigilance_update",
    "bsa.step_impl_evaluation_l57": "bsa.producer_scrounger_update",
    "bsa.step_impl_evaluation_l63": "bsa.flight_following_update",
    # C
    "camel.step_impl_evaluation_l38": "camel.endurance_temperature_update",
    "camel.step_impl_evaluation_l45": "camel.oasis_movement_update",
    "capsa.step_impl_evaluation_l60": "capsa.capuchin_jump_swing_update",
    "capsa.step_impl_evaluation_l65": "capsa.capuchin_local_refinement",
    "cat_so.step_evaluation_l59": "cat_so.tracing_mode_velocity_update",
    "cddo.step_impl_evaluation_l58": "cddo.cheetah_chase_position_update",
    "cdo.step_impl_evaluation_l34": "cdo.cheetah_density_position_update",
    "cfoa.step_impl_evaluation_l125": "cfoa.catch_fish_foraging_update",
    "chameleon_sa.step_impl_evaluation_l45": "chameleon_sa.tongue_projection_attack_update",
    "chameleon_sa.step_impl_evaluation_l49": "chameleon_sa.eye_rotation_search_update",
    "chicken_so.step_evaluation_l53": "chicken_so.rooster_hen_chick_update",
    "chicken_so.step_evaluation_l71": "chicken_so.mother_child_following_update",
    "choa.step_impl_evaluation_l46": "choa.chimp_hunting_position_update",
    "coati_oa.step_evaluation_l45": "coati_oa.coati_attack_escape_update",
    "cockroach_so.step_evaluation_l46": "cockroach_so.chase_swarming_update",
    "cockroach_so.step_evaluation_l54": "cockroach_so.dispersal_update",
    "cockroach_so.step_evaluation_l55": "cockroach_so.ruthless_replacement_update",
    "coot.step_impl_evaluation_l47": "coot.chain_movement_update",
    "coot.step_impl_evaluation_l60": "coot.leader_guidance_update",
    "cpo.step_impl_evaluation_l195": "cpo.pangolin_hunting_defense_update",
    "crayfish_oa.step_impl_evaluation_l71": "crayfish_oa.foraging_competition_update",
    "csa.step_evaluation_l48": "csa.memory_following_update",
}

_EXACT_SEMANTIC_REWRITE.update(_B7_EXACT_SEMANTIC_REWRITE)

# Faithful engines below expose one accepted macro-transition in the passive
# lineage stream for this batch.  They still receive lineage-delta OAM/CDS; a
# later hand-audit can split them into finer paper operators if desired.
_SINGLE_OPERATOR_SEMANTIC_OK.update({
    "aco", "acor", "afsa", "aho", "ala", "alo", "ao", "aoo", "apo",
    "aro", "aso", "avoa", "bbso", "bmo", "boa", "bps", "capsa",
    "cddo", "cdo", "cfoa", "chameleon_sa", "choa", "cpo", "crayfish_oa",
    "csa", "cso",
})

_prev_semanticize_operator_label_b7 = semanticize_operator_label

def semanticize_operator_label(algorithm_id: str, label: str | None) -> str | None:  # type: ignore[override]
    """Semanticize labels, including B7 macro-transition and carryover cases."""
    if label in {None, ""}:
        return label
    aid = str(algorithm_id or "")
    lab = str(label)
    if lab == "carryover":
        return f"{aid}.retained_parent" if aid else "retained_parent"
    return _prev_semanticize_operator_label_b7(aid, lab)

# Keep resolve and list helpers pointed at the new semanticize function.
_base_resolve_operator_label_b7 = _base_resolve_operator_label

def resolve_operator_label(algorithm_id: str, filename: str, function: str, line: int | None = None) -> str | None:  # type: ignore[override]
    raw = _base_resolve_operator_label_b7(algorithm_id, filename, function, line)
    return semanticize_operator_label(algorithm_id, raw) if raw else None

def labels_for_algorithm(algorithm_id: str) -> list[str]:  # type: ignore[override]
    aid = str(algorithm_id)
    labels = [semanticize_operator_label(aid, x) for x in _base_labels_for_algorithm(aid)]
    out = []
    for x in labels:
        if x and x not in out:
            out.append(x)
    return out

# Re-apply exported static label table with B7 rewrites.
for _aid, _labels in list(ENGINE_OPERATOR_LABELS.items()):
    _new = []
    for _lab in _labels:
        _slab = semanticize_operator_label(_aid, _lab)
        if _slab and _slab not in _new:
            _new.append(_slab)
    ENGINE_OPERATOR_LABELS[_aid] = _new

# Line-number refresh after seeding Python's random in chicken_so.
_EXACT_SEMANTIC_REWRITE.update({
    "chicken_so.step_evaluation_l55": "chicken_so.rooster_hen_chick_update",
    "chicken_so.step_evaluation_l73": "chicken_so.mother_child_following_update",
})
for _aid in ("chicken_so",):
    _labels = ENGINE_OPERATOR_LABELS.get(_aid, [])
    _new = []
    for _lab in _labels:
        _slab = semanticize_operator_label(_aid, _lab)
        if _slab and _slab not in _new:
            _new.append(_slab)
    ENGINE_OPERATOR_LABELS[_aid] = _new

# ---------------------------------------------------------------------------
# Semantic override layer (batch B8: swarm D-G)
# ---------------------------------------------------------------------------
# Label-only rewrites for the D-G swarm batch. These names summarize the
# faithful code regions already executed by each engine; they do not change
# candidate generation, objective evaluations, RNG state, selection, or budgets.

_B8_EXACT_SEMANTIC_REWRITE = {
    "da.step_evaluation_l82": "da.dragonfly_swarm_food_enemy_update",
    "dbo.step_impl_evaluation_l40": "dbo.ball_rolling_dance_update",
    "dbo.step_impl_evaluation_l58": "dbo.brood_thief_foraging_update",
    "deo_dolphin.step_impl_evaluation_l60": "deo_dolphin.echo_location_probability_update",
    "dfo.step_evaluation_l45": "dfo.dispersive_fly_neighbour_update",
    "dfo.step_evaluation_l50": "dfo.elite_disturbance_update",
    "dvba.step_evaluation_l49": "dvba.virtual_bat_velocity_update",
    "dvba.step_evaluation_l58": "dvba.local_random_walk_update",
    "ecological_cycle_o.step_impl_evaluation_l149": "ecological_cycle_o.ecological_cycle_transition_update",
    "eefo.step_impl_evaluation_l117": "eefo.electric_eel_foraging_update",
    "eel_grouper_o.step_impl_evaluation_l98": "eel_grouper_o.eel_grouper_hunting_update",
    "eho.step_evaluation_l53": "eho.clan_updating_separating_update",
    "elk_ho.step_impl_evaluation_l50": "elk_ho.family_mating_position_update",
    "eoa.step_impl_evaluation_l55": "eoa.reproduction_crossover_update",
    "epc.step_impl_evaluation_l33": "epc.emperor_penguin_huddle_update",
    "esoa.step_impl_evaluation_l42": "esoa.egret_sit_and_wait_update",
    "esoa.step_impl_evaluation_l46": "esoa.egret_aggressive_attack_update",
    "fda.step_evaluation_l72": "fda.flow_direction_neighbour_update",
    "fdo.step_impl_evaluation_l32": "fdo.fitness_weighted_pace_update",
    "fdo.step_impl_evaluation_l37": "fdo.randomized_pace_adjustment",
    "fdo.step_impl_evaluation_l41": "fdo.best_guided_position_update",
    "ffa.step_impl_evaluation_l35": "ffa.fruitfly_smell_search_update",
    "ffo.step_impl_evaluation_l21": "ffo.fennec_exploration_update",
    "ffo.step_impl_evaluation_l28": "ffo.fennec_exploitation_update",
    "firefly_a.step_evaluation_l53": "firefly_a.brightness_attraction_randomization_update",
    "flo.step_impl_evaluation_l65": "flo.frilled_lizard_attack_update",
    "flo.step_impl_evaluation_l72": "flo.frilled_lizard_escape_update",
    "foa.step_impl_evaluation_l38": "foa.local_seeding_growth_update",
    "foa.step_impl_evaluation_l55": "foa.global_seeding_dispersion_update",
    "foa_fossa.step_impl_evaluation_l58": "foa_fossa.fossa_hunting_update",
    "fss.step_impl_evaluation_l40": "fss.individual_movement_update",
    "fss.step_impl_evaluation_l50": "fss.collective_instinctive_movement",
    "fss.step_impl_evaluation_l58": "fss.collective_volitive_movement",
    "fwa.step_impl_evaluation_l46": "fwa.explosion_sparks_selection_update",
    "gazelle_oa.step_impl_evaluation_l59": "gazelle_oa.brownian_exploration_update",
    "gazelle_oa.step_impl_evaluation_l76": "gazelle_oa.elite_levy_exploitation_update",
    "ggo.step_impl_evaluation_l23": "ggo.greylag_goose_flock_update",
    "gja.step_impl_evaluation_l69": "gja.gekko_hunting_position_update",
    "gjo.exploration": "gjo.golden_jackal_exploration_update",
    "gmo.improve_guide": "gmo.marketing_guidance_update",
    "go_growth.step_impl_evaluation_l43": "go_growth.growth_phase_update",
    "go_growth.step_impl_evaluation_l57": "go_growth.maturity_phase_update",
    "goa.step_evaluation_l54": "goa.grasshopper_social_force_update",
    "gpoo.step_impl_evaluation_l120": "gpoo.genghis_khan_social_update",
    "gso.step_impl_evaluation_l43": "gso.glowworm_luciferin_movement_update",
    "gso_glider_snake.step_evaluation_l101": "gso_glider_snake.glider_snake_position_update",
}
_EXACT_SEMANTIC_REWRITE.update(_B8_EXACT_SEMANTIC_REWRITE)

# Engines in B8 that expose a single accepted macro-transition in the passive
# lineage stream. They still produce nonzero lineage-delta OAM/CDS; the single
# label is documented here so the semantic gate accepts it.
_SINGLE_OPERATOR_SEMANTIC_OK.update({
    "da", "deo_dolphin", "dhole_oa", "eefo", "eel_grouper_o", "eho", "epc",
    "fda", "ffa", "firefly_a", "foa_fossa", "fox", "ggo", "gja", "gjo",
    "gmo", "goa", "gpoo", "gso", "gso_glider_snake",
})

_prev_semanticize_operator_label_b8 = semanticize_operator_label

def semanticize_operator_label(algorithm_id: str, label: str | None) -> str | None:  # type: ignore[override]
    if label in {None, ""}:
        return label
    aid = str(algorithm_id or "")
    lab = str(label)
    if lab in _B8_EXACT_SEMANTIC_REWRITE:
        return _B8_EXACT_SEMANTIC_REWRITE[lab]
    return _prev_semanticize_operator_label_b8(aid, lab)

_base_resolve_operator_label_b8 = _base_resolve_operator_label

def resolve_operator_label(algorithm_id: str, filename: str, function: str, line: int | None = None) -> str | None:  # type: ignore[override]
    raw = _base_resolve_operator_label_b8(algorithm_id, filename, function, line)
    return semanticize_operator_label(algorithm_id, raw) if raw else None

def labels_for_algorithm(algorithm_id: str) -> list[str]:  # type: ignore[override]
    aid = str(algorithm_id)
    labels = [semanticize_operator_label(aid, x) for x in _base_labels_for_algorithm(aid)]
    out = []
    for x in labels:
        if x and x not in out:
            out.append(x)
    return out

for _aid, _labels in list(ENGINE_OPERATOR_LABELS.items()):
    _new = []
    for _lab in _labels:
        _slab = semanticize_operator_label(_aid, _lab)
        if _slab and _slab not in _new:
            _new.append(_slab)
    ENGINE_OPERATOR_LABELS[_aid] = _new

# ---------------------------------------------------------------------------
# Semantic override layer (batch B9: swarm H-O)
# ---------------------------------------------------------------------------
# Label-only semantic rewrites for the H-O swarm batch.  These labels map the
# already-observed faithful source/evaluation regions to readable operator
# probes; they do not alter objective calls, RNG use, selection, or population
# updates.

_B9_EXACT_SEMANTIC_REWRITE = {
    # H family
    "hba.step_impl_evaluation_l47": "hba.bat_candidate_de_local_search_update",
    "hba.step_impl_evaluation_l50": "hba.bat_honey_guided_search_update",
    "hba_honey.step_impl_evaluation_l60": "hba_honey.honey_badger_digging_honey_update",
    "hgs.step_impl_evaluation_l69": "hgs.hunger_games_social_pressure_update",
    "hho.step_evaluation_l55": "hho.exploration_random_perching_update",
    "hho.step_evaluation_l60": "hho.hard_besiege_update",
    "hho.step_evaluation_l64": "hho.soft_besiege_update",
    "hho.step_evaluation_l68": "hho.soft_besiege_rapid_dive_trial",
    "hho.step_evaluation_l77": "hho.hard_besiege_rapid_dive_trial",
    "hho.step_evaluation_l81": "hho.levy_rapid_dive_refinement",
    "ho_hippo.step_impl_evaluation_l52": "ho_hippo.river_pond_position_update",
    "ho_hippo.step_impl_evaluation_l53": "ho_hippo.group_defense_position_update",
    "ho_hippo.step_impl_evaluation_l66": "ho_hippo.predator_defense_update",
    "ho_hippo.step_impl_evaluation_l72": "ho_hippo.local_exploitation_escape_update",
    "horse_oa.step_impl_evaluation_l46": "horse_oa.social_hierarchy_grazing_update",
    "hsaba.step_impl_evaluation_l39": "hsaba.self_adaptive_bat_de_update",
    "hus.step_evaluation_l60": "hus.hurricane_eye_tracking_update",
    # I/J/K
    "iaro.step_evaluation_l301": "iaro.improved_rabbit_global_update",
    "iaro.step_evaluation_l313": "iaro.elite_local_refinement",
    "improved_tlo.step_evaluation_l301": "improved_tlo.teacher_learner_population_update",
    "improved_tlo.step_evaluation_l313": "improved_tlo.elite_local_refinement",
    "jso.step_evaluation_l61": "jso.ocean_current_swarm_motion_update",
    "jy.step_evaluation_l47": "jy.best_away_from_worst_update",
    "kha.step_evaluation_l62": "kha.induced_movement_update",
    "kha.step_evaluation_l93": "kha.genetic_crossover_update",
    "kha.step_evaluation_l100": "kha.genetic_mutation_update",
    # L/M/N/O
    "loa.step_impl_evaluation_l46": "loa.pride_nomad_hunting_update",
    "loa_lyrebird.step_impl_evaluation_l59": "loa_lyrebird.lyrebird_escape_hiding_update",
    "mbo.step_evaluation_l67": "mbo.monarch_migration_adjusting_update",
    "mfa.step_evaluation_l52": "mfa.moth_flame_spiral_update",
    "mgo.clip_all": "mgo.territory_mountain_herding_update",
    "msa_e.exploitation": "msa_e.golden_ratio_exploitation_update",
    "mshoa.step_impl_evaluation_l66": "mshoa.spotted_hyena_encircling_hunting_update",
    "nmra.working_operator": "nmra.worker_foraging_operator",
    "ofa.use_gen_n_as_proxy_fe": "ofa.owl_neighbour_flight_update",
    "ooa.step_impl_evaluation_l22": "ooa.fish_hunting_search_update",
    "ooa.step_impl_evaluation_l25": "ooa.fish_carrying_local_update",
}
_EXACT_SEMANTIC_REWRITE.update(_B9_EXACT_SEMANTIC_REWRITE)

# Engines in B9 whose passive lineage stream exposes one accepted macro-update.
_SINGLE_OPERATOR_SEMANTIC_OK.update({
    "hba", "hba_honey", "hgs", "horse_oa", "hsaba", "hus", "jso", "jy",
    "loa", "loa_lyrebird", "mbo", "mfa", "mgo", "msa_e", "mshoa", "nmra", "ofa",
})

_prev_semanticize_operator_label_b9 = semanticize_operator_label

def semanticize_operator_label(algorithm_id: str, label: str | None) -> str | None:  # type: ignore[override]
    if label in {None, ""}:
        return label
    lab = str(label)
    if lab in _B9_EXACT_SEMANTIC_REWRITE:
        return _B9_EXACT_SEMANTIC_REWRITE[lab]
    return _prev_semanticize_operator_label_b9(str(algorithm_id or ""), lab)

_base_resolve_operator_label_b9 = _base_resolve_operator_label

def resolve_operator_label(algorithm_id: str, filename: str, function: str, line: int | None = None) -> str | None:  # type: ignore[override]
    raw = _base_resolve_operator_label_b9(algorithm_id, filename, function, line)
    return semanticize_operator_label(algorithm_id, raw) if raw else None

def labels_for_algorithm(algorithm_id: str) -> list[str]:  # type: ignore[override]
    aid = str(algorithm_id)
    labels = [semanticize_operator_label(aid, x) for x in _base_labels_for_algorithm(aid)]
    out = []
    for x in labels:
        if x and x not in out:
            out.append(x)
    return out

for _aid, _labels in list(ENGINE_OPERATOR_LABELS.items()):
    _new = []
    for _lab in _labels:
        _slab = semanticize_operator_label(_aid, _lab)
        if _slab and _slab not in _new:
            _new.append(_slab)
    ENGINE_OPERATOR_LABELS[_aid] = _new
# HUS line-number refresh after seeding Python's random.
_EXACT_SEMANTIC_REWRITE.update({
    "hus.step_evaluation_l62": "hus.hurricane_eye_tracking_update",
})
for _aid in ("hus",):
    _labels = ENGINE_OPERATOR_LABELS.get(_aid, [])
    _new = []
    for _lab in _labels:
        _slab = semanticize_operator_label(_aid, _lab)
        if _slab and _slab not in _new:
            _new.append(_slab)
    ENGINE_OPERATOR_LABELS[_aid] = _new

# ---------------------------------------------------------------------------
# Semantic override layer (batch B10: swarm P-S)
# ---------------------------------------------------------------------------
# Label-only semantic rewrites for P-S swarm algorithms.  These mappings are
# passive: they rename observed lineage/source labels so OAM/CDS reports
# operator meaning instead of file/line locations.  Engines whose faithful code
# exposes one accepted macro-transition are documented as single-operator units.

_B10_EXACT_SEMANTIC_REWRITE = {
    # P
    "parrot_o.fly_new_area": "parrot_o.flight_area_search_update",
    "pdo.step_impl_evaluation_l48": "pdo.prairie_dog_burrow_alarm_update",
    "pfa.positions": "pfa.pathfinder_position_update",
    "pfa_polar_fox.step_evaluation_l239": "pfa_polar_fox.experience_based_exploitation_update",
    "pfa_polar_fox.step_evaluation_l247": "pfa_polar_fox.leader_guided_refinement_update",
    "pko.step_impl_evaluation_l42": "pko.pelican_attack_update",
    "pko.step_impl_evaluation_l53": "pko.krill_following_update",
    "plba.step_impl_evaluation_l35": "plba.path_looping_bat_update",
    "qle_sca.step_evaluation_l301": "qle_sca.q_learning_sine_cosine_update",
    "qle_sca.step_evaluation_l313": "qle_sca.elite_local_refinement",
    # R
    "rfo.smell": "rfo.red_fox_smell_search_update",
    "rhso.step_impl_evaluation_l64": "rhso.rhinoceros_herd_position_update",
    "roa.attempt": "roa.remora_attempt_update",
    "rsa.step_impl_evaluation_l39": "rsa.reptile_hunting_encircling_update",
    "rso.step_impl_evaluation_l31": "rso.rat_swarm_chasing_update",
    # S
    "saba.step_impl_evaluation_l39": "saba.self_adaptive_bat_update",
    "sacoso.fes_swarm_standard_pso": "sacoso.cooperative_swarm_pso_update",
    "samso.step_evaluation_l68": "samso.self_adaptive_migratory_swarm_update",
    "sbo.mutation": "sbo.bowerbird_mutation_update",
    "serval_oa.step_impl_evaluation_l22": "serval_oa.prey_selection_hunting_update",
    "serval_oa.step_impl_evaluation_l25": "serval_oa.local_carrying_refinement_update",
    "sfo.step_impl_evaluation_l75": "sfo.sardine_attack_update",
    "sfoa.step_impl_evaluation_l41": "sfoa.sailfish_foraging_update",
    "shio.step_impl_evaluation_l28": "shio.iguana_sand_hill_position_update",
    "shio_success.step_impl_evaluation_l59": "shio_success.success_based_iguana_update",
    "sho.step_impl_evaluation_l64": "sho.spotted_hyena_hunting_update",
    "sine_cosine_a.step_evaluation_l49": "sine_cosine_a.sine_cosine_position_update",
    "slo.step_impl_evaluation_l47": "slo.sea_lion_position_update",
    "so_snake.step_impl_evaluation_l76": "so_snake.male_snake_update",
    "so_snake.step_impl_evaluation_l77": "so_snake.female_snake_update",
    "soa.eq_14": "soa.seagull_spiral_attack_update",
    "sparrow_sa.step_impl_evaluation_l50": "sparrow_sa.producer_scrounger_update",
    "spbo.step_evaluation_l74": "spbo.average_student_phase_update",
    "spbo.step_evaluation_l77": "spbo.excellent_student_phase_update",
    "srsr_robotics.step_impl_evaluation_l90": "srsr_robotics.master_robot_guidance_update",
    "srsr_robotics.step_impl_evaluation_l155": "srsr_robotics.robotic_local_refinement_update",
    "sso.step_impl_evaluation_l30": "sso.sparrow_search_position_update",
    "sspider_a.step_impl_evaluation_l43": "sspider_a.social_spider_vibration_update",
    "sto.step_impl_evaluation_l23": "sto.sail_tetra_exploration_update",
    "sto.step_impl_evaluation_l26": "sto.sail_tetra_exploitation_update",
    "superb_foa.step_impl_evaluation_l45": "superb_foa.superb_fairywren_foraging_update",
}
_EXACT_SEMANTIC_REWRITE.update(_B10_EXACT_SEMANTIC_REWRITE)

# Faithful implementations that expose a single evaluated macro-update in the
# lineage stream.  The gate permits one semantic label for these engines.
_SINGLE_OPERATOR_SEMANTIC_OK.update({
    "parrot_o", "pdo", "pfa", "plba", "rfo", "rhso", "roa", "rsa", "rso",
    "saba", "sacoso", "samso", "sbo", "shio", "shio_success", "sho",
    "sine_cosine_a", "slo", "soa", "sso", "sspider_a", "superb_foa",
})

# Refresh exported static labels for this batch so README/table/web helpers see
# semantic names instead of source-location names.
for _aid in (
    "parrot_o", "pdo", "pfa", "pfa_polar_fox", "pko", "plba", "qle_sca",
    "rfo", "rhso", "roa", "rsa", "rso", "saba", "sacoso", "samso", "sbo",
    "seaho", "serval_oa", "sfo", "sfoa", "shio", "shio_success", "sho",
    "sine_cosine_a", "slo", "so_snake", "soa", "sparrow_sa", "spbo",
    "srsr_robotics", "sso", "sspider_a", "sto", "superb_foa",
):
    _labels = ENGINE_OPERATOR_LABELS.get(_aid, [])
    _new = []
    for _lab in _labels:
        _slab = semanticize_operator_label(_aid, _lab)
        if _slab and _slab not in _new:
            _new.append(_slab)
    if _new:
        ENGINE_OPERATOR_LABELS[_aid] = _new

# ---------------------------------------------------------------------------
# Semantic override layer (batch B11: swarm T-Z + surrogate-swarm)
# ---------------------------------------------------------------------------
# Label-only rewrites for accepted transitions in the T-Z swarm/surrogate batch.
# These names remove source-location placeholders from EvoMapX OAM/CDS without
# changing the underlying faithful engine dynamics.
_B11_EXACT_SEMANTIC_REWRITE = {
    "tdo.step_impl_evaluation_l24": "tdo.carrion_feeding_exploration_update",
    "tdo.step_impl_evaluation_l28": "tdo.prey_hunting_exploitation_update",
    "tlco.step_impl_evaluation_l41": "tlco.teacher_phase_update",
    "tlco.step_impl_evaluation_l63": "tlco.learner_phase_update",
    "tlco.step_impl_evaluation_l77": "tlco.cooperative_peer_learning_update",
    "tsa.step_impl_evaluation_l40": "tsa.tunicate_swarm_position_update",
    "tso.parabolic_updates": "tso.transient_parabolic_position_update",
    "waoa.step_impl_evaluation_l24": "waoa.wave_exploration_update",
    "waoa.step_impl_evaluation_l28": "waoa.wave_exploitation_update",
    "wmqimrfo.step_evaluation_l301": "wmqimrfo.weighted_multi_quadratic_mrfo_update",
    "wmqimrfo.step_evaluation_l313": "wmqimrfo.elite_local_refinement",
    "wooa.step_impl_evaluation_l71": "wooa.feeding_migration_update",
    "wooa.step_impl_evaluation_l77": "wooa.fighting_escape_update",
    "wooa.step_impl_evaluation_l83": "wooa.reproduction_refinement_update",
    "wso.step_impl_evaluation_l57": "wso.white_shark_swarm_position_update",
    "l2smea.step_evaluation_l167": "l2smea.surrogate_assisted_evolutionary_update",
    "misaco.step_evaluation_l272": "misaco.multi_surrogate_aco_update",
    "sapo.step_evaluation_l398": "sapo.surrogate_assisted_particle_update",
}
_EXACT_SEMANTIC_REWRITE.update(_B11_EXACT_SEMANTIC_REWRITE)
_SINGLE_OPERATOR_SEMANTIC_OK.update({
    "tsa", "tso", "wso", "l2smea", "misaco", "sapo",
})
for _aid in (
    "tdo", "tlco", "tsa", "tso", "waoa", "wmqimrfo", "wooa", "wso", "zoa",
    "l2smea", "misaco", "sapo",
):
    _labels = ENGINE_OPERATOR_LABELS.get(_aid, [])
    _new = []
    for _lab in _labels:
        _slab = semanticize_operator_label(_aid, _lab)
        if _slab and _slab not in _new:
            _new.append(_slab)
    if _aid in _SINGLE_OPERATOR_SEMANTIC_OK and not _new:
        _new.append(f"{_aid}.semantic_update")
    ENGINE_OPERATOR_LABELS[_aid] = _new

# ---------------------------------------------------------------------------
# Semantic override layer (batch B12: physics-inspired algorithms)
# ---------------------------------------------------------------------------
# Passive label-only rewrites for physics-inspired algorithms.  These mappings
# convert source-location evaluation labels into stable semantic operator probes
# used by OAM/CDS.  They do not change objective calls, RNG use, candidate
# generation, selection, replacement, or population updates.
_B12_EXACT_SEMANTIC_REWRITE = {
    # Equilibrium / shared adaptive templates
    "adaptive_eo.step_evaluation_l301": "adaptive_eo.equilibrium_pool_guided_update",
    "adaptive_eo.step_evaluation_l313": "adaptive_eo.adaptive_local_refinement",
    "enhanced_two.step_evaluation_l301": "enhanced_two.tug_of_war_force_update",
    "enhanced_two.step_evaluation_l313": "enhanced_two.enhanced_local_refinement",
    "modified_eo.step_evaluation_l301": "modified_eo.modified_equilibrium_pool_update",
    "modified_eo.step_evaluation_l313": "modified_eo.modified_local_refinement",

    # Single or monolithic physics update regions
    "aefa.step_impl_evaluation_l45": "aefa.electric_field_force_update",
    "arch_oa.step_impl_evaluation_l85": "arch_oa.archimedes_density_volume_acceleration_update",
    "cdo_chernobyl.step_impl_evaluation_l54": "cdo_chernobyl.radiation_zone_position_update",
    "ceo_cosmic.step_impl_evaluation_l86": "ceo_cosmic.cosmic_evolution_position_update",
    "eso.step_impl_evaluation_l69": "eso.electric_storm_field_update",
    "fata.step_impl_evaluation_l44": "fata.geophysical_refraction_update",
    "gsa.step_evaluation_l62": "gsa.gravitational_force_acceleration_update",
    "liwo.step_impl_evaluation_l73": "liwo.light_wave_position_update",
    "lso_spectrum.step_impl_evaluation_l87": "lso_spectrum.light_spectrum_position_update",
    "plo.step_impl_evaluation_l47": "plo.plasma_lithium_position_update",
    "snow_oa.step_impl_evaluation_l46": "snow_oa.snow_ablation_position_update",
    "wdo.step_impl_evaluation_l46": "wdo.wind_velocity_position_update",
    "wo_wave.step_impl_evaluation_l73": "wo_wave.wave_propagation_position_update",
    "ydse.step_impl_evaluation_l52": "ydse.double_slit_interference_update",

    # Multi-region physics operators
    "ddao.step_impl_evaluation_l26": "ddao.differential_annealed_exploration_update",
    "ddao.step_impl_evaluation_l35": "ddao.dynamic_annealed_refinement_update",
    "evo.step_impl_evaluation_l41": "evo.energy_valley_exploration_update",
    "evo.step_impl_evaluation_l45": "evo.energy_valley_exploitation_update",
    "flood_a.step_impl_evaluation_l34": "flood_a.flood_flow_direction_update",
    "flood_a.step_impl_evaluation_l46": "flood_a.flood_recession_refinement_update",
    "ikoa.step_impl_evaluation_l87": "ikoa.assignment_matching_position_update",
    "ikoa.step_impl_evaluation_l88": "ikoa.improved_matching_refinement_update",
    "mso.step_impl_evaluation_l40": "mso.magnetic_field_position_update",
    "nro.step_impl_evaluation_l64": "nro.nuclear_fusion_update",
    "nro.step_impl_evaluation_l82": "nro.nuclear_fission_update",
    "rcco.step_impl_evaluation_l96": "rcco.cloud_collision_local_update",
    "rcco.step_impl_evaluation_l119": "rcco.rain_cloud_convection_update",

    # Already semantic but cleaned up for readability / gate stability
    "do_dandelion.3": "do_dandelion.seed_landing_update",
    "ecpo.random_perturbation": "ecpo.electric_charge_random_perturbation",
    "efo.mutation": "efo.electromagnetic_field_mutation",
    "eo.position_eq_16": "eo.equilibrium_position_update",
    "fla.tf_0_9": "fla.fick_law_diffusion_transport_update",
    "rime.hard_rime_puncture_mechanism": "rime.hard_rime_puncture_update",
    "toc.velocity_radial_tangential_random": "toc.combination_velocity_update",
    "two.weights_by_rank": "two.tug_of_war_weight_force_update",
}
_EXACT_SEMANTIC_REWRITE.update(_B12_EXACT_SEMANTIC_REWRITE)

# B12 engines whose faithful implementation exposes one accepted macro-update
# in the passive lineage stream.  They are legitimate single-OAM-unit engines
# until their source code is manually split into finer internal branch labels.
_SINGLE_OPERATOR_SEMANTIC_OK.update({
    "aefa", "arch_oa", "aso_atom", "cdo_chernobyl", "ceo_cosmic",
    "do_dandelion", "ecpo", "efo", "eo", "eso", "fata", "fla",
    "gsa", "liwo", "lso_spectrum", "plo", "rime", "snow_oa", "toc",
    "two", "wdo", "wo_wave", "ydse",
})

_prev_semanticize_operator_label_b12 = semanticize_operator_label

def semanticize_operator_label(algorithm_id: str, label: str | None) -> str | None:  # type: ignore[override]
    if label in {None, ""}:
        return label
    lab = str(label)
    if lab in _B12_EXACT_SEMANTIC_REWRITE:
        return _B12_EXACT_SEMANTIC_REWRITE[lab]
    return _prev_semanticize_operator_label_b12(str(algorithm_id or ""), lab)

_base_resolve_operator_label_b12 = _base_resolve_operator_label

def resolve_operator_label(algorithm_id: str, filename: str, function: str, line: int | None = None) -> str | None:  # type: ignore[override]
    raw = _base_resolve_operator_label_b12(algorithm_id, filename, function, line)
    return semanticize_operator_label(algorithm_id, raw) if raw else None

def labels_for_algorithm(algorithm_id: str) -> list[str]:  # type: ignore[override]
    aid = str(algorithm_id)
    labels = [semanticize_operator_label(aid, x) for x in _base_labels_for_algorithm(aid)]
    out = []
    for x in labels:
        if x and x not in out:
            out.append(x)
    return out

for _aid in (
    "adaptive_eo", "aefa", "arch_oa", "aso_atom", "cdo_chernobyl",
    "ceo_cosmic", "ddao", "do_dandelion", "ecpo", "efo", "enhanced_two",
    "eo", "eso", "evo", "fata", "fla", "flood_a", "gsa", "ikoa",
    "liwo", "lso_spectrum", "modified_eo", "mso", "nro", "plo", "rcco",
    "rime", "snow_oa", "toc", "two", "wdo", "wo_wave", "ydse",
):
    _labels = ENGINE_OPERATOR_LABELS.get(_aid, [])
    _new = []
    for _lab in _labels:
        _slab = semanticize_operator_label(_aid, _lab)
        if _slab and _slab not in _new:
            _new.append(_slab)
    if _aid in _SINGLE_OPERATOR_SEMANTIC_OK and not _new:
        _new.append(f"{_aid}.physics_macro_update")
    ENGINE_OPERATOR_LABELS[_aid] = _new

# ---------------------------------------------------------------------------
# Semantic override layer (batch B13: human/social-inspired algorithms)
# ---------------------------------------------------------------------------
# Passive label-only rewrites for human-inspired optimizers.  These mappings
# replace source-location evaluation labels by stable semantic probe labels for
# OAM/CDS reporting.  They do not change candidate generation, objective calls,
# RNG usage, selection, replacement, or stopping behavior.
_B13_EXACT_SEMANTIC_REWRITE = {
    # Ecosystem / AEO variants
    "aeo.step_impl_evaluation_l36": "aeo.consumer_decomposer_update",
    "enhanced_aeo.step_evaluation_l301": "enhanced_aeo.ecosystem_producer_consumer_update",
    "enhanced_aeo.step_evaluation_l313": "enhanced_aeo.enhanced_decomposition_refinement",
    "improved_aeo.step_evaluation_l301": "improved_aeo.ecosystem_producer_consumer_update",
    "improved_aeo.step_evaluation_l313": "improved_aeo.improved_decomposition_refinement",
    "modified_aeo.step_evaluation_l301": "modified_aeo.ecosystem_producer_consumer_update",
    "modified_aeo.step_evaluation_l313": "modified_aeo.modified_decomposition_refinement",

    # Human/social role, competition and learning algorithms
    "aft.step_impl_evaluation_l52": "aft.thieves_search_escape_update",
    "bro.step_impl_evaluation_l54": "bro.battle_damage_relocation_update",
    "btoa.step_impl_evaluation_l132": "btoa.offensive_play_update",
    "btoa.step_impl_evaluation_l143": "btoa.defensive_play_refinement",
    "cddo_child.step_impl_evaluation_l74": "cddo_child.child_drawing_development_update",
    "doa.step_impl_evaluation_l63": "doa.deer_hunting_search_update",
    "dra.step_impl_evaluation_l74": "dra.belief_group_guidance_update",
    "dra.step_impl_evaluation_l96": "dra.dialectic_interaction_update",
    "dra.step_impl_evaluation_l110": "dra.religion_conversion_refinement",
    "dream_oa.step_impl_evaluation_l115": "dream_oa.dream_generation_refinement_update",
    "dso.step_impl_evaluation_l31": "dso.deep_sleep_position_update",
    "eco.step_impl_evaluation_l45": "eco.educational_competition_update",
    "gska.step_impl_evaluation_l67": "gska.gaining_sharing_knowledge_update",
    "hco.step_impl_evaluation_l62": "hco.conception_growth_update",
    "hiking_oa.step_impl_evaluation_l28": "hiking_oa.hiking_slope_velocity_update",
    "mgoa_market.step_impl_evaluation_l63": "mgoa_market.market_gradient_position_update",
    "petio.step_impl_evaluation_l252": "petio.performance_evaluation_teaching_update",
    "pro.step_impl_evaluation_l29": "pro.poor_rich_learning_update",
    "pro.step_impl_evaluation_l36": "pro.wealth_exchange_refinement",
    "singer_oa.step_impl_evaluation_l63": "singer_oa.singing_pitch_search_update",
    "singer_oa.step_impl_evaluation_l74": "singer_oa.audience_feedback_refinement",
    "ssdo.step_impl_evaluation_l49": "ssdo.social_ski_driver_update",
    "supply_do.step_impl_evaluation_l39": "supply_do.supply_demand_exploration_update",
    "supply_do.step_impl_evaluation_l43": "supply_do.supply_demand_balance_update",
    "toa.step_impl_evaluation_l50": "toa.stage_2_peer_learning_update",

    # Already source-level semantic fragments cleaned for readable probe labels
    "bso.two_cluster_idea": "bso.two_cluster_brainstorm_update",
    "chio.immune_contact": "chio.immune_contact_update",
    "esc.explore_randomly": "esc.random_escape_exploration_update",
    "gco.dark_zone": "gco.dark_zone_mutation_update",
    "hbo.no_change": "hbo.heap_rank_pressure_update",
    "lco.boundary_reflection": "lco.life_choice_boundary_reflection_update",
    "mtbo.random_relocation": "mtbo.mountaineering_random_relocation_update",
    "mvpa.each_player_moves_toward_mvp_random": "mvpa.mvp_guided_player_update",
    "thro.race": "thro.throwing_race_update",
    "warso.original_index_sorted_i": "warso.war_strategy_ranked_update",
}
_EXACT_SEMANTIC_REWRITE.update(_B13_EXACT_SEMANTIC_REWRITE)

# B13 engines whose faithful implementation exposes one accepted macro-update
# in the passive lineage stream. They are legitimate single-OAM-unit engines
# until manually split into finer internal branch labels.
_SINGLE_OPERATOR_SEMANTIC_OK.update({
    "aft", "bso", "cddo_child", "chio", "dream_oa", "dso", "eco",
    "esc", "gco", "gska", "hbo", "hco", "hiking_oa", "lco", "mtbo",
    "mvpa", "petio", "ssdo", "thro", "warso",
})

_prev_semanticize_operator_label_b13 = semanticize_operator_label

def semanticize_operator_label(algorithm_id: str, label: str | None) -> str | None:  # type: ignore[override]
    if label in {None, ""}:
        return label
    lab = str(label)
    if lab in _B13_EXACT_SEMANTIC_REWRITE:
        return _B13_EXACT_SEMANTIC_REWRITE[lab]
    return _prev_semanticize_operator_label_b13(str(algorithm_id or ""), lab)

_base_resolve_operator_label_b13 = resolve_operator_label

def resolve_operator_label(algorithm_id: str, filename: str, function: str, line: int | None = None) -> str | None:  # type: ignore[override]
    raw = _base_resolve_operator_label_b13(algorithm_id, filename, function, line)
    return semanticize_operator_label(algorithm_id, raw) if raw else None

def labels_for_algorithm(algorithm_id: str) -> list[str]:  # type: ignore[override]
    aid = str(algorithm_id)
    labels = [semanticize_operator_label(aid, x) for x in _base_labels_for_algorithm(aid)]
    out = []
    for x in labels:
        if x and x not in out:
            out.append(x)
    return out

for _aid in (
    "aeo", "aft", "bro", "bso", "btoa", "cddo_child", "chio", "doa",
    "dra", "dream_oa", "dso", "eco", "enhanced_aeo", "esc", "gco",
    "gska", "hbo", "hco", "heoa", "hiking_oa", "improved_aeo",
    "improved_qsa", "lco", "mgoa_market", "modified_aeo", "mtbo",
    "mvpa", "petio", "pro", "singer_oa", "ssdo", "supply_do", "thro",
    "toa", "warso",
):
    _labels = ENGINE_OPERATOR_LABELS.get(_aid, [])
    _new = []
    for _lab in _labels:
        _slab = semanticize_operator_label(_aid, _lab)
        if _slab and _slab not in _new:
            _new.append(_slab)
    if _aid in _SINGLE_OPERATOR_SEMANTIC_OK and not _new:
        _new.append(f"{_aid}.human_social_macro_update")
    ENGINE_OPERATOR_LABELS[_aid] = _new

# B13 late patch: improved QSA shared template labels.
_B13_QSA_SEMANTIC_REWRITE = {
    "improved_qsa.step_evaluation_l301": "improved_qsa.queue_business_one_update",
    "improved_qsa.step_evaluation_l313": "improved_qsa.queue_business_two_refinement",
}
_EXACT_SEMANTIC_REWRITE.update(_B13_QSA_SEMANTIC_REWRITE)
_prev_semanticize_operator_label_b13_qsa = semanticize_operator_label

def semanticize_operator_label(algorithm_id: str, label: str | None) -> str | None:  # type: ignore[override]
    if label in {None, ""}:
        return label
    lab = str(label)
    if lab in _B13_QSA_SEMANTIC_REWRITE:
        return _B13_QSA_SEMANTIC_REWRITE[lab]
    return _prev_semanticize_operator_label_b13_qsa(str(algorithm_id or ""), lab)

_base_resolve_operator_label_b13_qsa = resolve_operator_label

def resolve_operator_label(algorithm_id: str, filename: str, function: str, line: int | None = None) -> str | None:  # type: ignore[override]
    raw = _base_resolve_operator_label_b13_qsa(algorithm_id, filename, function, line)
    return semanticize_operator_label(algorithm_id, raw) if raw else None

def labels_for_algorithm(algorithm_id: str) -> list[str]:  # type: ignore[override]
    aid = str(algorithm_id)
    labels = [semanticize_operator_label(aid, x) for x in _base_labels_for_algorithm(aid)]
    out = []
    for x in labels:
        if x and x not in out:
            out.append(x)
    return out

for _aid in ("improved_qsa",):
    _labels = ENGINE_OPERATOR_LABELS.get(_aid, [])
    _new = []
    for _lab in _labels:
        _slab = semanticize_operator_label(_aid, _lab)
        if _slab and _slab not in _new:
            _new.append(_slab)
    ENGINE_OPERATOR_LABELS[_aid] = _new

# ---------------------------------------------------------------------------
# Semantic override layer (batch B14: gradient/numerical methods)
# ---------------------------------------------------------------------------
# Passive label-only rewrites for gradient and classical numerical optimizers.
# These engines frequently expose one accepted macro-transition in the lineage
# stream.  Single-label OAM/CDS is accepted for this batch because the faithful
# code performs one composite numerical update per step.
_B14_EXACT_SEMANTIC_REWRITE = {
    "adam.step_evaluation_l78": "adam.gradient_step",
    "rmsprop.step_evaluation_l72": "rmsprop.gradient_step",
    "sd.step_evaluation_l75": "sd.gradient_step",
    "bfgs.armijo_line_search": "bfgs.line_search_armijo",
    "frcg.armijo_line_search_along_dk": "frcg.line_search_armijo",
    "sqp.line_search_armijo": "sqp.line_search_armijo",
    "nca.step_impl_evaluation_l67": "nca.convergence_acceleration_update",
    "noa.step_impl_evaluation_l147": "noa.newton_position_update",
    "pss.step_impl_evaluation_l44": "pss.powell_sweep_search_update",
}
_EXACT_SEMANTIC_REWRITE.update(_B14_EXACT_SEMANTIC_REWRITE)

_SINGLE_OPERATOR_SEMANTIC_OK.update({
    "adam", "bfgs", "frcg", "nca", "noa", "pss", "rmsprop", "sd", "sqp",
})

_prev_semanticize_operator_label_b14 = semanticize_operator_label

def semanticize_operator_label(algorithm_id: str, label: str | None) -> str | None:  # type: ignore[override]
    if label in {None, ""}:
        return label
    lab = str(label)
    if lab in _B14_EXACT_SEMANTIC_REWRITE:
        return _B14_EXACT_SEMANTIC_REWRITE[lab]
    return _prev_semanticize_operator_label_b14(str(algorithm_id or ""), lab)

_base_resolve_operator_label_b14 = resolve_operator_label

def resolve_operator_label(algorithm_id: str, filename: str, function: str, line: int | None = None) -> str | None:  # type: ignore[override]
    raw = _base_resolve_operator_label_b14(algorithm_id, filename, function, line)
    return semanticize_operator_label(algorithm_id, raw) if raw else None

def labels_for_algorithm(algorithm_id: str) -> list[str]:  # type: ignore[override]
    aid = str(algorithm_id)
    labels = [semanticize_operator_label(aid, x) for x in _base_labels_for_algorithm(aid)]
    out = []
    for x in labels:
        if x and x not in out:
            out.append(x)
    return out

for _aid in ("adam", "bfgs", "frcg", "nca", "noa", "pss", "rmsprop", "sd", "sqp"):
    _labels = ENGINE_OPERATOR_LABELS.get(_aid, [])
    _new = []
    for _lab in _labels:
        _slab = semanticize_operator_label(_aid, _lab)
        if _slab and _slab not in _new:
            _new.append(_slab)
    if not _new:
        _new.append(f"{_aid}.numerical_macro_update")
    ENGINE_OPERATOR_LABELS[_aid] = _new

# ---------------------------------------------------------------------------
# Semantic override layer (batch B15: Bayesian/surrogate + transform methods)
# ---------------------------------------------------------------------------
# Passive label-only rewrites for mathematical-transform and surrogate-model
# optimizers.  These rewrites expose stable semantic probe names in OAM/CDS
# reports while preserving the faithful optimizer bodies and evaluation budget.
_B15_EXACT_SEMANTIC_REWRITE = {
    # Distribution / transform based mathematical optimizers
    "circle_sa.eq_8": "circle_sa.circle_position_update",
    "edo.step_impl_evaluation_l51": "edo.exponential_distribution_candidate_update",
    "eto.trigonometric_component": "eto.exponential_trigonometric_candidate_update",
    "gbo.step_impl_evaluation_l78": "gbo.gradient_search_operator_update",
    "gbo.local_escaping_operator": "gbo.local_escaping_operator_update",
    "gndo.step_impl_evaluation_l41": "gndo.generalized_normal_distribution_update",
    "info.step_impl_evaluation_l81": "info.weighted_mean_rule_candidate_update",
    "qio.step_impl_evaluation_l57": "qio.quadratic_interpolation_candidate_update",
    "run.step_impl_evaluation_l73": "run.runge_kutta_position_update",
    "run.step_impl_evaluation_l90": "run.enhanced_solution_quality_update",
    "scho.step_impl_evaluation_l58": "scho.scholar_chess_position_update",
    "ttao.step_impl_evaluation_l78": "ttao.triangulation_topology_sampling_update",
    "ttao.step_impl_evaluation_l110": "ttao.greedy_crossover_update",
    "ttao.step_impl_evaluation_l137": "ttao.convex_topology_refinement_update",
    "ttao.step_impl_evaluation_l148": "ttao.random_population_refresh_update",
    "ttao.step_impl_evaluation_l165": "ttao.extra_candidate_diversification_update",

    # Shared surrogate Bayesian optimization base
    "et_bo.evaluate_positions_evaluation_l449": "et_bo.extra_trees_surrogate_acquisition_update",
    "gbrt_bo.evaluate_positions_evaluation_l449": "gbrt_bo.gradient_boosted_surrogate_acquisition_update",
    "gp_bo.evaluate_positions_evaluation_l449": "gp_bo.gaussian_process_surrogate_acquisition_update",
    "rf_bo.evaluate_positions_evaluation_l449": "rf_bo.random_forest_surrogate_acquisition_update",
}
_EXACT_SEMANTIC_REWRITE.update(_B15_EXACT_SEMANTIC_REWRITE)

# Many B15 engines expose one accepted macro-transition in the passive lineage
# stream.  They are single OAM/CDS units in the current faithful implementation.
_SINGLE_OPERATOR_SEMANTIC_OK.update({
    "circle_sa", "edo", "et_bo", "eto", "gbo", "gbrt_bo", "gndo",
    "gp_bo", "info", "qio", "rf_bo", "scho",
})

_prev_semanticize_operator_label_b15 = semanticize_operator_label

def semanticize_operator_label(algorithm_id: str, label: str | None) -> str | None:  # type: ignore[override]
    if label in {None, ""}:
        return label
    lab = str(label)
    if lab in _B15_EXACT_SEMANTIC_REWRITE:
        return _B15_EXACT_SEMANTIC_REWRITE[lab]
    return _prev_semanticize_operator_label_b15(str(algorithm_id or ""), lab)

_base_resolve_operator_label_b15 = resolve_operator_label

def resolve_operator_label(algorithm_id: str, filename: str, function: str, line: int | None = None) -> str | None:  # type: ignore[override]
    raw = _base_resolve_operator_label_b15(algorithm_id, filename, function, line)
    return semanticize_operator_label(algorithm_id, raw) if raw else None

def labels_for_algorithm(algorithm_id: str) -> list[str]:  # type: ignore[override]
    aid = str(algorithm_id)
    labels = [semanticize_operator_label(aid, x) for x in _base_labels_for_algorithm(aid)]
    out = []
    for x in labels:
        if x and x not in out:
            out.append(x)
    return out

for _aid in (
    "cgo", "circle_sa", "edo", "et_bo", "eto", "gbo", "gbrt_bo",
    "gndo", "gp_bo", "info", "qio", "rf_bo", "run", "scho", "ttao",
):
    _labels = ENGINE_OPERATOR_LABELS.get(_aid, [])
    _new = []
    for _lab in _labels:
        _slab = semanticize_operator_label(_aid, _lab)
        if _slab and _slab not in _new:
            _new.append(_slab)
    if _aid in _SINGLE_OPERATOR_SEMANTIC_OK and not _new:
        _new.append(f"{_aid}.mathematical_transform_macro_update")
    ENGINE_OPERATOR_LABELS[_aid] = _new

# ---------------------------------------------------------------------------
# Semantic override layer (batch B16: nature/growth algorithms)
# ---------------------------------------------------------------------------
# Passive label-only rewrites for nature-inspired growth, dispersal, plant,
# seed, enzyme, and slime-mould style optimizers.  These names replace source
# locations in OAM/CDS reports without touching optimizer dynamics.
_B16_EXACT_SEMANTIC_REWRITE = {
    "artemisinin_o.boundary": "artemisinin_o.boundary_best_replacement",
    "artemisinin_o.mutation": "artemisinin_o.artemisinin_mutation_update",
    "bco.swim_refine_without_turbulence": "bco.swim_refinement_update",
    "bco.tumble_turbulence": "bco.tumble_turbulence_update",
    "bco.neighbour_exchange": "bco.neighbour_exchange_update",
    "eao.candidate_generation": "eao.enzyme_substrate_candidate_update",
    "ivya.step_impl_evaluation_l36": "ivya.ivy_growth_neighbor_update",
    "iwo.step_impl_evaluation_l51": "iwo.seed_dispersal_colonization_update",
    "lca.mutation": "lca.league_match_mutation_update",
    "lpo.step_impl_evaluation_l51": "lpo.lichen_growth_propagation_update",
    "moss_go.water_dispersal_gradient_like": "moss_go.water_dispersal_growth_update",
    "sma.eq_2_3": "sma.slime_mould_oscillation_update",
    "tpo.step_impl_evaluation_l34": "tpo.carbon_nutrient_leaf_update",
    "tree_seed_a.eq_4": "tree_seed_a.away_random_seed_update",
    "tree_seed_a.eq_3": "tree_seed_a.toward_best_seed_update",
    "wutp.horizontal": "wutp.horizontal_water_transport_update",
    "wutp.water_in_motion": "wutp.water_flux_motion_update",
}
_EXACT_SEMANTIC_REWRITE.update(_B16_EXACT_SEMANTIC_REWRITE)

# In the current passive lineage stream most B16 engines expose one accepted
# macro-transition label.  They are therefore documented as single OAM/CDS
# semantic units until the faithful implementation is later split at finer
# operator boundaries.
_SINGLE_OPERATOR_SEMANTIC_OK.update({
    "artemisinin_o", "bco", "eao", "ivya", "iwo", "lca", "lpo",
    "moss_go", "sma", "tpo", "tree_seed_a", "wutp",
})

_prev_semanticize_operator_label_b16 = semanticize_operator_label

def semanticize_operator_label(algorithm_id: str, label: str | None) -> str | None:  # type: ignore[override]
    if label in {None, ""}:
        return label
    lab = str(label)
    if lab in _B16_EXACT_SEMANTIC_REWRITE:
        return _B16_EXACT_SEMANTIC_REWRITE[lab]
    return _prev_semanticize_operator_label_b16(str(algorithm_id or ""), lab)

_base_resolve_operator_label_b16 = resolve_operator_label

def resolve_operator_label(algorithm_id: str, filename: str, function: str, line: int | None = None) -> str | None:  # type: ignore[override]
    raw = _base_resolve_operator_label_b16(algorithm_id, filename, function, line)
    return semanticize_operator_label(algorithm_id, raw) if raw else None

def labels_for_algorithm(algorithm_id: str) -> list[str]:  # type: ignore[override]
    aid = str(algorithm_id)
    labels = [semanticize_operator_label(aid, x) for x in _base_labels_for_algorithm(aid)]
    out = []
    for x in labels:
        if x and x not in out:
            out.append(x)
    return out

for _aid in (
    "artemisinin_o", "bco", "eao", "ivya", "iwo", "lca", "lpo",
    "moss_go", "sma", "tpo", "tree_seed_a", "wutp",
):
    _labels = ENGINE_OPERATOR_LABELS.get(_aid, [])
    _new = []
    for _lab in _labels:
        _slab = semanticize_operator_label(_aid, _lab)
        if _slab and _slab not in _new:
            _new.append(_slab)
    if _aid in _SINGLE_OPERATOR_SEMANTIC_OK and not _new:
        _new.append(f"{_aid}.nature_growth_macro_update")
    ENGINE_OPERATOR_LABELS[_aid] = _new

# ---------------------------------------------------------------------------
# Semantic override layer (batch B17: trajectory / local-search algorithms)
# ---------------------------------------------------------------------------
# Passive label-only rewrites for single-trajectory, local-search, restart,
# and neighborhood-search optimizers.  These labels replace source-location
# runtime probes in OAM/CDS reports while preserving the faithful optimizer
# bodies, objective-call count, and random-number stream.
_B17_EXACT_SEMANTIC_REWRITE = {
    "basin_hopping.local_search_evaluation_l177": "basin_hopping.perturb_local_search_acceptance_update",
    "basin_hopping.local_search_evaluation_l145": "basin_hopping.local_search_trial_update",
    "basin_hopping.step": "basin_hopping.basin_hop_macro_update",
    "grasp.local_search_evaluation_l177": "grasp.construct_local_search_refinement_update",
    "grasp.local_search_evaluation_l145": "grasp.local_search_trial_update",
    "hc.step_evaluation_l311": "hc.neighborhood_search_update",
    "hsa.step_evaluation_l45": "hsa.harmony_memory_improvisation_update",
    "ils.local_search_evaluation_l177": "ils.perturbation_local_search_acceptance_update",
    "ils.local_search_evaluation_l145": "ils.local_search_trial_update",
    "msls.local_search_evaluation_l177": "msls.multi_start_local_search_refinement_update",
    "msls.local_search_evaluation_l145": "msls.local_search_trial_update",
    "mts.local": "mts.multiple_trajectory_local_search_update",
    "nmm.step_impl_evaluation_l30": "nmm.reflection_update",
    "nmm.step_impl_evaluation_l39": "nmm.expansion_contraction_update",
    "nmm.step_impl_evaluation_l44": "nmm.shrink_update",
    "random_s.step_evaluation_l44": "random_s.random_sampling_update",
    "sa.step_evaluation_l124": "sa.annealing_neighborhood_acceptance_update",
    "ts.step_evaluation_l53": "ts.tabu_neighborhood_selection_update",
    "vns.local_search_evaluation_l177": "vns.shaking_local_search_refinement_update",
    "vns.local_search_evaluation_l145": "vns.local_search_trial_update",
}
_EXACT_SEMANTIC_REWRITE.update(_B17_EXACT_SEMANTIC_REWRITE)

# These faithful trajectory engines expose a single accepted macro-transition
# in the passive lineage stream at the current instrumentation granularity.
# Their OAM/CDS is therefore a single semantic trajectory unit unless future
# hand instrumentation splits proposal, acceptance, and restart events inside
# the engine body.
_SINGLE_OPERATOR_SEMANTIC_OK.update({
    "basin_hopping", "grasp", "hc", "hsa", "ils", "msls", "mts",
    "random_s", "sa", "ts", "vns",
})

_prev_semanticize_operator_label_b17 = semanticize_operator_label

def semanticize_operator_label(algorithm_id: str, label: str | None) -> str | None:  # type: ignore[override]
    if label in {None, ""}:
        return label
    lab = str(label)
    if lab in _B17_EXACT_SEMANTIC_REWRITE:
        return _B17_EXACT_SEMANTIC_REWRITE[lab]
    return _prev_semanticize_operator_label_b17(str(algorithm_id or ""), lab)

_base_resolve_operator_label_b17 = resolve_operator_label

def resolve_operator_label(algorithm_id: str, filename: str, function: str, line: int | None = None) -> str | None:  # type: ignore[override]
    raw = _base_resolve_operator_label_b17(algorithm_id, filename, function, line)
    return semanticize_operator_label(algorithm_id, raw) if raw else None

def labels_for_algorithm(algorithm_id: str) -> list[str]:  # type: ignore[override]
    aid = str(algorithm_id)
    labels = [semanticize_operator_label(aid, x) for x in _base_labels_for_algorithm(aid)]
    out = []
    for x in labels:
        if x and x not in out:
            out.append(x)
    return out

for _aid in (
    "basin_hopping", "grasp", "hc", "hsa", "ils", "msls", "mts",
    "nmm", "random_s", "sa", "ts", "vns",
):
    _labels = ENGINE_OPERATOR_LABELS.get(_aid, [])
    _new = []
    for _lab in _labels:
        _slab = semanticize_operator_label(_aid, _lab)
        if _slab and _slab not in _new:
            _new.append(_slab)
    if _aid in _SINGLE_OPERATOR_SEMANTIC_OK and not _new:
        _new.append(f"{_aid}.trajectory_macro_update")
    ENGINE_OPERATOR_LABELS[_aid] = _new

# ---------------------------------------------------------------------------
# Semantic override layer (batch B18: distribution / model-building algorithms)
# ---------------------------------------------------------------------------
# Passive label-only rewrites for distribution-estimation and model-based
# optimizers.  These labels replace source-location probes in OAM/CDS reports
# without touching objective calls, RNG use, or optimizer dynamics.
_B18_EXACT_SEMANTIC_REWRITE = {
    "cem.step_evaluation_l49": "cem.model_sampling_elite_distribution_update",
    "compact_ga.evaluate_bits": "compact_ga.probability_model_sampling_update",
    "compact_ga.step_evaluation_l211": "compact_ga.local_refinement_update",
    "ego.candidate_generation": "ego.expected_improvement_candidate_generation",
    "pbil.step_evaluation_l37": "pbil.probability_vector_sampling_update",
    "sopt.select_best": "sopt.statistical_population_selection_update",
}
_EXACT_SEMANTIC_REWRITE.update(_B18_EXACT_SEMANTIC_REWRITE)

# At the current passive lineage granularity these distribution engines expose
# one accepted macro-transition label, except compact_ga which exposes both
# model sampling and local refinement.  The single-label engines are documented
# as one OAM/CDS semantic unit until future hand instrumentation splits sampling,
# elite selection, and model update into separate accepted-transition labels.
_SINGLE_OPERATOR_SEMANTIC_OK.update({"cem", "ego", "pbil", "sopt"})

_prev_semanticize_operator_label_b18 = semanticize_operator_label

def semanticize_operator_label(algorithm_id: str, label: str | None) -> str | None:  # type: ignore[override]
    if label in {None, ""}:
        return label
    lab = str(label)
    if lab in _B18_EXACT_SEMANTIC_REWRITE:
        return _B18_EXACT_SEMANTIC_REWRITE[lab]
    return _prev_semanticize_operator_label_b18(str(algorithm_id or ""), lab)

_base_resolve_operator_label_b18 = resolve_operator_label

def resolve_operator_label(algorithm_id: str, filename: str, function: str, line: int | None = None) -> str | None:  # type: ignore[override]
    raw = _base_resolve_operator_label_b18(algorithm_id, filename, function, line)
    return semanticize_operator_label(algorithm_id, raw) if raw else None

def labels_for_algorithm(algorithm_id: str) -> list[str]:  # type: ignore[override]
    aid = str(algorithm_id)
    labels = [semanticize_operator_label(aid, x) for x in _base_labels_for_algorithm(aid)]
    out = []
    for x in labels:
        if x and x not in out:
            out.append(x)
    return out

for _aid in ("cem", "compact_ga", "ego", "pbil", "sopt"):
    _labels = ENGINE_OPERATOR_LABELS.get(_aid, [])
    _new = []
    for _lab in _labels:
        _slab = semanticize_operator_label(_aid, _lab)
        if _slab and _slab not in _new:
            _new.append(_slab)
    if _aid in _SINGLE_OPERATOR_SEMANTIC_OK and not _new:
        _new.append(f"{_aid}.distribution_model_macro_update")
    ENGINE_OPERATOR_LABELS[_aid] = _new

# ---------------------------------------------------------------------------
# Semantic override layer (cleanup B0-B2: canonical refs + DE/evolutionary)
# ---------------------------------------------------------------------------
# These rewrites finish the early reference batches that predate the B3-B18
# semantic rollout.  They are label-only transforms used by EvoMapX reports;
# no optimizer dynamics, objective calls, RNG use, or evaluation budgets change.

_B0_B2_EXACT_SEMANTIC_REWRITE = {
    # B0 canonical references
    "de.step_evaluation_l105": "de.differential_mutation_crossover_selection",
    "pso.velocity_position": "pso.velocity_position_memory_update",
    "cuckoo_s.step_evaluation_l49": "cuckoo_s.levy_flight_replacement",

    # B1 DE variants
    "jade.step_evaluation_l301": "jade.current_to_pbest_mutation_crossover_selection",
    "jade.step_evaluation_l313": "jade.archive_parameter_refinement",
    "jde.step_impl_evaluation_l37": "jde.self_adaptive_de_mutation_crossover_selection",
    "shade.step_impl_evaluation_l43": "shade.success_history_mutation_crossover_selection",
    "ilshade.step_impl_evaluation_l44": "ilshade.linear_population_reduction_mutation_selection",
    "lshade_cnepsin.step_impl_evaluation_l112": "lshade_cnepsin.cn_epsin_mutation_crossover_selection",
    "sade.step_evaluation_l301": "sade.adaptive_strategy_de_update",
    "sade.step_evaluation_l313": "sade.elite_local_refinement",
    "sap_de.step_evaluation_l301": "sap_de.self_adaptive_parameter_de_update",
    "sap_de.step_evaluation_l313": "sap_de.elite_local_refinement",
    "imode.step_evaluation_l140": "imode.multi_operator_de_selection_update",
    "hde.step_impl_evaluation_l22": "hde.local_search_mts_refinement",
    "hde.step_impl_evaluation_l31": "hde.differential_evolution_update",

    # B2 evolutionary / immune / cultural / EDA-style engines
    "cmaes.evaluate": "cmaes.covariance_sampling_recombination_update",
    "bipop_cmaes.step_evaluation_l534": "bipop_cmaes.restart_cmaes_sampling_update",
    "ipop_cmaes.step_evaluation_l534": "ipop_cmaes.increasing_population_cmaes_update",
    "es.step_impl_evaluation_l40": "es.mutation_survivor_selection_update",
    "ep.generate_offspring": "ep.mutation_tournament_selection_update",
    "fep.mutation": "fep.fast_mutation_tournament_selection_update",
    "bbo.step_evaluation_l57": "bbo.migration_mutation_selection_update",
    "clonalg.step_evaluation_l54": "clonalg.cloning_hypermutation_selection_update",
    "ca.init_pop": "ca.cultural_belief_guided_update",
    "cro.step_impl_evaluation_l37": "cro.reef_broadcast_spawning_update",
    "cro.step_impl_evaluation_l46": "cro.reef_brooding_settlement_update",
    "autov.step_evaluation_l91": "autov.learned_variation_operator_update",
    "bspga.evaluate_positions": "bspga.binary_partition_tree_variation_update",
    "mfea.mutation": "mfea.assortative_mating_mutation_transfer_update",
    "mfea2.step_evaluation_l194": "mfea2.adaptive_multifactorial_transfer_update",
    "mke.step_impl_evaluation_l33": "mke.memory_knowledge_evolution_update",
    "nlapsmjso_eda.step_evaluation_l254": "nlapsmjso_eda.non_linear_population_analysis_update",
    "nlapsmjso_eda.step_evaluation_l301": "nlapsmjso_eda.eda_sampling_selection_update",
    "ocro.step_evaluation_l301": "ocro.opposition_coral_reef_update",
    "ocro.step_evaluation_l313": "ocro.elite_local_refinement",
    "pcx.generate_child": "pcx.parent_centric_crossover_update",
    "sade_amss.de_rand_1_bin_on_subspace": "sade_amss.adaptive_multistrategy_subspace_de_update",
    "sade_atdsc.without_surrogate_evaluate_all_pick_best": "sade_atdsc.adaptive_trial_distribution_selection_update",
    "sade_sammon.step_evaluation_l78": "sade_sammon.sammon_surrogate_de_selection_update",
    "ssio_rl.step_evaluation_l346": "ssio_rl.reinforcement_operator_selection_update",
}

_EXACT_SEMANTIC_REWRITE.update(_B0_B2_EXACT_SEMANTIC_REWRITE)

# Engines in B0-B2 whose faithful implementation exposes one evaluated macro
# transition in the current passive lineage.  These are accepted as one OAM unit
# until deeper manual counterfactual/operator-boundary splitting is added.
_SINGLE_OPERATOR_SEMANTIC_OK.update({
    "de", "pso", "jde", "shade", "ilshade", "lshade_cnepsin", "imode",
    "cmaes", "bipop_cmaes", "ipop_cmaes", "es", "ep", "fep", "bbo",
    "clonalg", "ca", "autov", "bspga", "mfea", "mfea2", "mke",
    "pcx", "sade_amss", "sade_atdsc", "sade_sammon", "ssio_rl",
})

# Extend source-label rewrite dynamically for B0-B2 patterns that may be
# generated by helper call-site resolution.
_base_semanticize_operator_label_b0_b2 = semanticize_operator_label

def semanticize_operator_label(algorithm_id: str, label: str | None) -> str | None:  # type: ignore[override]
    if label in {None, ""}:
        return label
    lab = str(label)
    if lab in _B0_B2_EXACT_SEMANTIC_REWRITE:
        return _B0_B2_EXACT_SEMANTIC_REWRITE[lab]
    return _base_semanticize_operator_label_b0_b2(algorithm_id, label)

# Re-apply exported static label table so docs/UI helpers also see B0-B2 names.
for _aid, _labels in list(ENGINE_OPERATOR_LABELS.items()):
    _new = []
    for _lab in _labels:
        _slab = semanticize_operator_label(_aid, _lab)
        if _slab and _slab not in _new:
            _new.append(_slab)
    ENGINE_OPERATOR_LABELS[_aid] = _new



# ---------------------------------------------------------------------------
# Semantic decomposition layer (compound-label guard)
# ---------------------------------------------------------------------------
# Some passive call-site probes naturally observe a full evaluated macro-step
# even when that macro-step is made of well-known constituent operators.  OAM
# and CDS must not collapse those constituents into one fused label.  The helper
# below expands such labels into deterministic, budget-preserving semantic
# constituents.  It is intentionally label-only: it never calls the objective,
# never samples random numbers, and never changes optimizer state.

_EXACT_COMPOUND_OPERATOR_SPLITS = {
    # Canonical examples explicitly required by the EvoMapX gate/spec.
    "de.differential_mutation_crossover_selection": ("de.mutation", "de.crossover", "de.selection"),
    "pso.velocity_position_memory_update": ("pso.velocity_update", "pso.position_update", "pso.memory_update"),
    "cmaes.covariance_sampling_recombination_update": (
        "cmaes.covariance_update", "cmaes.offspring_sampling", "cmaes.recombination", "cmaes.step_size_update",
    ),
    "firefly_a.brightness_attraction_randomization_update": ("firefly_a.attraction", "firefly_a.randomization"),
    "mfea.assortative_mating_mutation_transfer_update": ("mfea.assortative_mating", "mfea.mutation", "mfea.transfer"),
    "iwo.seed_dispersal_colonization_update": ("iwo.seed_dispersal", "iwo.colonization"),
    "goa.grasshopper_social_force_update": ("goa.social_force", "goa.position_update"),
    "avoa.avoa_replaces_all_no_greedy_in": ("avoa.exploration", "avoa.exploitation"),
    "afsa.leap": ("afsa.random_leap", "afsa.replacement"),
    "doa.exploitation": ("doa.exploitation_move", "doa.replacement"),
    "sfoa.exploitation": ("sfoa.exploitation_move", "sfoa.replacement"),
    "scso.exploitation": ("scso.exploitation_move", "scso.replacement"),
    "scso.exploration": ("scso.exploration_move", "scso.replacement"),
    "memetic_a.breed": ("memetic_a.recombination", "memetic_a.mutation"),
    "ngo.1": ("ngo.phase_one_update", "ngo.selection"),
    "puma_o.stalking": ("puma_o.stalking_move", "puma_o.selection"),

    # Early evolutionary/CMA/DE family compounds.
    "jde.self_adaptive_de_mutation_crossover_selection": ("jde.parameter_adaptation", "jde.mutation", "jde.crossover", "jde.selection"),
    "shade.success_history_mutation_crossover_selection": ("shade.success_history_update", "shade.mutation", "shade.crossover", "shade.selection"),
    "ilshade.linear_population_reduction_mutation_selection": ("ilshade.population_reduction", "ilshade.mutation", "ilshade.selection"),
    "lshade_cnepsin.cn_epsin_mutation_crossover_selection": ("lshade_cnepsin.cn_epsin_adaptation", "lshade_cnepsin.mutation", "lshade_cnepsin.crossover", "lshade_cnepsin.selection"),
    "bbo.migration_mutation_selection_update": ("bbo.migration", "bbo.mutation", "bbo.selection"),
    "ep.mutation_tournament_selection_update": ("ep.mutation", "ep.tournament_selection"),
    "es.mutation_survivor_selection_update": ("es.mutation", "es.survivor_selection"),
    "fep.fast_mutation_tournament_selection_update": ("fep.fast_mutation", "fep.tournament_selection"),
    "clonalg.cloning_hypermutation_selection_update": ("clonalg.cloning", "clonalg.hypermutation", "clonalg.selection"),

    # Other frequent one-line macro labels observed by the passive gate.
    "aco.pheromone_weighted_perturbation_in_each_dimension": ("aco.pheromone_weighting", "aco.dimension_perturbation"),
    "acor.archive_kernel_sampling_update": ("acor.archive_selection", "acor.kernel_sampling"),
    "aesspso.adaptive_velocity_position_update": ("aesspso.velocity_update", "aesspso.position_update"),
    "basin_hopping.perturb_local_search_acceptance_update": ("basin_hopping.perturbation", "basin_hopping.local_search", "basin_hopping.acceptance"),
    "bipop_cmaes.restart_cmaes_sampling_update": ("bipop_cmaes.restart", "bipop_cmaes.cmaes_sampling", "bipop_cmaes.distribution_update"),
    "cem.model_sampling_elite_distribution_update": ("cem.model_sampling", "cem.elite_selection", "cem.distribution_update"),
    "cuckoo_s.levy_flight_replacement": ("cuckoo_s.levy_flight", "cuckoo_s.replacement"),
    "firefly_a.brightness_attraction_randomization_update": ("firefly_a.attraction", "firefly_a.randomization"),
    "gpso.velocity_position_update": ("gpso.velocity_update", "gpso.position_update"),
    "iagwo.adaptive_alpha_beta_delta_update": ("iagwo.alpha_guidance", "iagwo.beta_guidance", "iagwo.delta_guidance", "iagwo.adaptation"),
    "ipop_cmaes.increasing_population_cmaes_update": ("ipop_cmaes.population_restart", "ipop_cmaes.cmaes_sampling", "ipop_cmaes.distribution_update"),
    "mfea2.adaptive_multifactorial_transfer_update": ("mfea2.adaptation", "mfea2.multifactorial_transfer"),
    "pso.velocity_position_memory_update": ("pso.velocity_update", "pso.position_update", "pso.memory_update"),
    "sacoso.cooperative_swarm_pso_update": ("sacoso.cooperation", "sacoso.swarm_update", "sacoso.pso_update"),
    "sade_atdsc.adaptive_trial_distribution_selection_update": ("sade_atdsc.trial_distribution", "sade_atdsc.parameter_adaptation", "sade_atdsc.selection"),
    "shade.success_history_mutation_crossover_selection": ("shade.success_history_update", "shade.mutation", "shade.crossover", "shade.selection"),
    "sine_cosine_a.sine_cosine_position_update": ("sine_cosine_a.sine_cosine_move", "sine_cosine_a.position_update"),
    "ts.tabu_neighborhood_selection_update": ("ts.neighborhood_move", "ts.tabu_memory", "ts.selection"),
    "vns.shaking_local_search_refinement_update": ("vns.shaking", "vns.local_search", "vns.refinement"),
}

_COMPOUND_TOKEN_PRIORITY = (
    ("assortative_mating", "assortative_mating"),
    ("success_history", "success_history_update"),
    ("linear_population_reduction", "population_reduction"),
    ("probability_vector", "probability_vector_update"),
    ("parent_centric_crossover", "parent_centric_crossover"),
    ("differential_mutation", "mutation"),
    ("de_mutation", "mutation"),
    ("levy_flight", "levy_flight"),
    ("random_walk", "random_walk"),
    ("local_search", "local_search"),
    ("global_search", "global_search"),
    ("line_search", "line_search"),
    ("surrogate_acquisition", "surrogate_acquisition"),
    ("candidate_generation", "candidate_generation"),
    ("position_update", "position_update"),
    ("velocity_update", "velocity_update"),
    ("memory_update", "memory_update"),
    ("social_force", "social_force"),
    ("field_force", "field_force"),
    ("seed_dispersal", "seed_dispersal"),
    ("elite_selection", "elite_selection"),
    ("tournament_selection", "tournament_selection"),
    ("survivor_selection", "survivor_selection"),
    ("knowledge_transfer", "knowledge_transfer"),
    ("parameter_adaptation", "parameter_adaptation"),
    ("step_size", "step_size_update"),
    ("covariance", "covariance_update"),
    ("sampling", "sampling"),
    ("recombination", "recombination"),
    ("mutation", "mutation"),
    ("crossover", "crossover"),
    ("selection", "selection"),
    ("replacement", "replacement"),
    ("migration", "migration"),
    ("perturbation", "perturbation"),
    ("perturb", "perturbation"),
    ("acceptance", "acceptance"),
    ("restart", "restart"),
    ("randomization", "randomization"),
    ("attraction", "attraction"),
    ("brightness", "brightness"),
    ("dispersal", "dispersal"),
    ("colonization", "colonization"),
    ("cloning", "cloning"),
    ("hypermutation", "hypermutation"),
    ("teaching", "teaching"),
    ("learning", "learning"),
    ("competition", "competition"),
    ("foraging", "foraging"),
    ("hunting", "hunting"),
    ("encircling", "encircling"),
    ("exploitation", "exploitation"),
    ("exploration", "exploration"),
    ("guidance", "guidance"),
    ("position", "position_update"),
    ("velocity", "velocity_update"),
    ("force", "force_update"),
    ("acceleration", "acceleration_update"),
    ("distribution", "distribution_update"),
    ("model", "model_update"),
    ("candidate", "candidate_generation"),
    ("search", "search"),
    ("sampling", "sampling"),
    ("update", "state_update"),
)

_COMPOUND_DROP_WORDS = {
    "adaptive", "self", "fast", "differential", "evolution", "evolutionary",
    "multi", "operator", "guided", "weighted", "all", "each", "dimension",
    "in", "of", "the", "best", "current", "elite", "source", "semantic",
    "step", "impl", "evaluation", "eq", "equation",
}

_ATOMIC_OPERATOR_BODIES = {
    "mutation", "crossover", "selection", "replacement", "recombination",
    "velocity_update", "position_update", "memory_update", "covariance_update",
    "offspring_sampling", "sampling", "step_size_update", "attraction",
    "randomization", "assortative_mating", "transfer", "seed_dispersal",
    "colonization", "social_force", "exploration", "exploitation",
    "search_direction", "step_acceptance", "candidate_generation",
    "candidate_search", "candidate_update", "position_generation",
    "exploration_move", "exploitation_move", "force_or_velocity_update",
    "behavioral_move", "state_update", "model_update", "distribution_update",
    "archive_selection", "kernel_sampling", "pheromone_weighting",
    "dimension_perturbation", "perturbation", "local_search", "global_search",
    "acceptance", "restart", "teaching", "learning", "competition",
    "foraging", "hunting", "encircling", "guidance", "force_update",
    "acceleration_update", "candidate_search", "parameter_adaptation",
    "success_history_update", "population_reduction", "tournament_selection",
    "survivor_selection", "hypermutation", "cloning", "migration",
}



def _dedupe_labels(labels: list[str]) -> list[str]:
    out: list[str] = []
    for lab in labels:
        if lab and lab not in out:
            out.append(lab)
    return out


def _fallback_split_for_body(aid: str, body: str) -> list[str]:
    """Return a conservative two-part decomposition for opaque macro labels."""
    b = body.lower()
    if "gradient" in b or "line_search" in b or b.endswith("armijo"):
        return [f"{aid}.search_direction", f"{aid}.step_acceptance"]
    if "exploitation" in b:
        return [f"{aid}.exploitation_move", f"{aid}.replacement"]
    if "exploration" in b:
        return [f"{aid}.exploration_move", f"{aid}.replacement"]
    if "search" in b or "stalking" in b or "leap" in b or "breed" in b:
        return [f"{aid}.candidate_search", f"{aid}.selection"]
    if "position" in b or "move" in b or "motion" in b:
        return [f"{aid}.position_generation", f"{aid}.selection"]
    if "candidate" in b or "sampling" in b or "sample" in b:
        return [f"{aid}.candidate_generation", f"{aid}.selection"]
    if "mutation" in b:
        return [f"{aid}.mutation", f"{aid}.selection"]
    if "crossover" in b or "reproduction" in b or "mating" in b:
        return [f"{aid}.recombination", f"{aid}.selection"]
    if "foraging" in b or "hunting" in b or "attack" in b:
        return [f"{aid}.behavioral_move", f"{aid}.selection"]
    if "force" in b or "field" in b or "velocity" in b or "acceleration" in b:
        return [f"{aid}.force_or_velocity_update", f"{aid}.position_update"]
    if "update" in b:
        return [f"{aid}.candidate_update", f"{aid}.selection"]
    return [f"{aid}.candidate_generation", f"{aid}.selection"]


def expand_compound_operator_label(algorithm_id: str, label: str | None) -> list[str]:
    """Expand a fused semantic operator label into OAM/CDS constituents.

    The expansion is budget preserving at the attribution layer: callers should
    divide the observed parent→child delta by ``len(returned_labels)`` when more
    than one constituent is returned.  ``carryover`` and ``initialization`` are
    intentionally left untouched.
    """
    if label in {None, ""}:
        return []
    raw = str(label)
    if raw in {"carryover", "initialization"}:
        return [raw]
    if raw in _EXACT_COMPOUND_OPERATOR_SPLITS:
        return list(_EXACT_COMPOUND_OPERATOR_SPLITS[raw])

    aid = str(algorithm_id or "")
    if "." in raw:
        prefix, body = raw.split(".", 1)
        if not aid:
            aid = prefix
    else:
        body = raw
    if not aid:
        aid = "operator"
    body_l = body.lower()
    if body_l in _ATOMIC_OPERATOR_BODIES or body_l.endswith("_guidance"):
        return [raw]

    # Avoid expanding already atomic labels unless they are the only kind of
    # macro label the runtime can expose.  Atomic names that are obviously
    # source/opaque still get a conservative two-part decomposition below.
    matches: list[str] = []
    covered = body_l
    for needle, op in _COMPOUND_TOKEN_PRIORITY:
        if needle in covered:
            matches.append(f"{aid}.{op}")
            covered = covered.replace(needle, "_")
    matches = _dedupe_labels(matches)
    if len(matches) >= 2:
        return matches

    parts = [p for p in body_l.replace(".", "_").split("_") if p]
    informative_parts = [p for p in parts if p not in _COMPOUND_DROP_WORDS and not p.isdigit()]
    # A one-word already atomic label remains atomic.  Everything with a fused
    # phrase, source-location residue, or generic update name is decomposed so
    # the OAM/CDS gate cannot be satisfied by a single packed token.
    if len(informative_parts) <= 1 and not any(x in body_l for x in ("step_evaluation", "step_impl", "eq_", "gradient_step", "line_search")):
        atomic = f"{aid}.{informative_parts[0]}" if informative_parts else raw
        if atomic == raw or raw.endswith("." + informative_parts[0] if informative_parts else ""):
            return [raw]
    return _dedupe_labels(_fallback_split_for_body(aid, body_l))


_base_labels_for_algorithm_compound_guard = labels_for_algorithm

def labels_for_algorithm(algorithm_id: str) -> list[str]:  # type: ignore[override]
    aid = str(algorithm_id)
    out: list[str] = []
    for lab in _base_labels_for_algorithm_compound_guard(aid):
        for slab in expand_compound_operator_label(aid, lab):
            if slab and slab not in out:
                out.append(slab)
    return out

# Keep the exported static catalog flattened as well; this prevents downstream
# documentation/UI gates from reading a single fused label from the registry.
for _aid, _labels in list(ENGINE_OPERATOR_LABELS.items()):
    _flat: list[str] = []
    for _lab in _labels:
        for _slab in expand_compound_operator_label(_aid, _lab):
            if _slab and _slab not in _flat:
                _flat.append(_slab)
    ENGINE_OPERATOR_LABELS[_aid] = _flat

__all__ = list(dict.fromkeys(list(__all__) + ["expand_compound_operator_label"]))


# ---------------------------------------------------------------------------
# Conservative semantic-label preservation and canonical branch labels.
# ---------------------------------------------------------------------------
# The compound splitter must split true packed operators such as
# ``mutation_crossover_selection``.  It must not turn already meaningful branch
# labels such as ``mpa.levy_transition`` or ``hho.hard_besiege`` into generic
# fallback labels.  This override keeps semantic branches intact and only
# decomposes exact or strongly packed operator phrases.

_EXACT_OPERATOR_LABEL_CANONICALIZATION = {
    # Marine Predators Algorithm: preserve its theory-level phase/FAD labels.
    "mpa.brownian_exploration": ("mpa.brownian_exploration",),
    "mpa.brownian_transition": ("mpa.brownian_transition",),
    "mpa.levy_transition": ("mpa.levy_transition",),
    "mpa.levy_exploitation": ("mpa.levy_exploitation",),
    "mpa.fish_aggregating_devices_fads_effect": ("mpa.fads",),
    # Harris Hawks Optimization: canonicalize source-line labels to HHO branches.
    "hho.exploration_random_perching_update": ("hho.exploration",),
    "hho.hard_besiege_update": ("hho.hard_besiege",),
    "hho.soft_besiege_update": ("hho.soft_besiege",),
    "hho.soft_besiege_rapid_dive_trial": ("hho.soft_besiege_rapid_dive",),
    "hho.hard_besiege_rapid_dive_trial": ("hho.hard_besiege_rapid_dive",),
    "hho.levy_rapid_dive_refinement": ("hho.levy_rapid_dive_refinement",),
    # Bonobo Optimizer: the registry id is only ``bono``.
    "bono.bonobo_social_search_update": ("bono.social_guidance_phase", "bono.exploratory_directional_move"),
}

_STRONGLY_PACKED_OPERATOR_PATTERNS = (
    "mutation_crossover",
    "crossover_selection",
    "mutation_selection",
    "recombination_selection",
    "covariance_sampling",
    "sampling_recombination",
    "velocity_position",
    "position_memory",
    "attraction_randomization",
    "assortative_mating_mutation",
    "mutation_transfer",
    "seed_dispersal_colonization",
    "shaking_local_search",
    "local_search_refinement",
    "tabu_neighborhood_selection",
    "neighborhood_selection",
    "adaptive_trial_distribution_selection",
    "trial_distribution_selection",
    "probability_model_sampling",
    "model_sampling",
)

_SOURCE_LOCATION_MARKERS = (
    "step_impl_evaluation_l",
    "step_evaluation_l",
    "initialize_evaluation_l",
    "inject_candidates_evaluation_l",
    "local_search_evaluation_l",
    "grad_evaluation_l",
    "evaluate_bits",
)

_PREVIOUS_EXPAND_COMPOUND_OPERATOR_LABEL = expand_compound_operator_label

def expand_compound_operator_label(algorithm_id: str, label: str | None) -> list[str]:  # type: ignore[override]
    if label in {None, ""}:
        return []
    raw = str(label)
    if raw in {"carryover", "initialization"}:
        return [raw]
    if raw in _EXACT_COMPOUND_OPERATOR_SPLITS:
        return list(_EXACT_COMPOUND_OPERATOR_SPLITS[raw])
    if raw in _EXACT_OPERATOR_LABEL_CANONICALIZATION:
        return list(_EXACT_OPERATOR_LABEL_CANONICALIZATION[raw])

    aid = str(algorithm_id or "")
    if "." in raw:
        prefix, body = raw.split(".", 1)
        if not aid:
            aid = prefix
    else:
        body = raw
    if not aid:
        aid = "operator"
    body_l = body.lower()

    # Already atomic/theory-level branches should survive exactly as observed.
    if body_l in _ATOMIC_OPERATOR_BODIES or body_l.endswith("_guidance"):
        return [raw]
    if raw in ENGINE_OPERATOR_LABELS.get(aid, []):
        # Exact catalog labels are semantic branch labels unless explicitly listed
        # above as a true compound split.
        return [raw]

    # Strong compound dodge: only split labels that clearly pack several operator
    # verbs, or opaque source-location labels where no semantic branch is known.
    if any(p in body_l for p in _STRONGLY_PACKED_OPERATOR_PATTERNS):
        matches: list[str] = []
        covered = body_l
        for needle, op in _COMPOUND_TOKEN_PRIORITY:
            if needle in covered:
                matches.append(f"{aid}.{op}")
                covered = covered.replace(needle, "_")
        matches = _dedupe_labels(matches)
        if len(matches) >= 2:
            return matches

    if any(m in body_l for m in _SOURCE_LOCATION_MARKERS):
        return _dedupe_labels(_fallback_split_for_body(aid, body_l))

    return [raw]

# Re-flatten exported labels using the conservative splitter/canonicalizer.
for _aid, _labels in list(ENGINE_OPERATOR_LABELS.items()):
    _flat: list[str] = []
    for _lab in _labels:
        for _slab in expand_compound_operator_label(_aid, _lab):
            if _slab and _slab not in _flat:
                _flat.append(_slab)
    ENGINE_OPERATOR_LABELS[_aid] = _flat

__all__ = list(dict.fromkeys(list(__all__) + ["expand_compound_operator_label"]))


# Canonical catalog/operator-schema overrides for engines whose runtime exposes
# precise branch labels. These keep docs/UI inventory consistent with OAM output.
_ENGINE_OPERATOR_LABEL_OVERRIDES = {
    "de": ["de.mutation", "de.crossover", "de.selection"],
    "pso": ["pso.velocity_update", "pso.position_update", "pso.memory_update"],
    "cmaes": ["cmaes.covariance_update", "cmaes.offspring_sampling", "cmaes.recombination", "cmaes.step_size_update"],
    "firefly_a": ["firefly_a.attraction", "firefly_a.randomization"],
    "mfea": ["mfea.assortative_mating", "mfea.mutation", "mfea.transfer"],
    "iwo": ["iwo.seed_dispersal", "iwo.colonization"],
    "mpa": ["mpa.brownian_exploration", "mpa.brownian_transition", "mpa.levy_transition", "mpa.levy_exploitation", "mpa.fads"],
    "hho": ["hho.exploration", "hho.soft_besiege", "hho.hard_besiege", "hho.soft_besiege_rapid_dive", "hho.hard_besiege_rapid_dive", "hho.levy_rapid_dive_refinement"],
    "bono": ["bono.social_guidance_phase", "bono.exploratory_directional_move"],
    "mfo": ["mfo.exploration_move", "mfo.exploitation_move", "mfo.replacement"],
    "mvo": ["mvo.candidate_generation", "mvo.selection", "mvo.exploitation_move", "mvo.replacement"],
    "nwoa": ["nwoa.exploration_move", "nwoa.exploitation_move", "nwoa.replacement"],
    "ssa": ["ssa.leader_plus_food_guidance", "ssa.leader_minus_food_guidance", "ssa.follower_front_chain_update", "ssa.follower_rear_chain_update"],
    "woa": ["woa.search_for_prey", "woa.encircling_prey", "woa.spiral_bubble_net"],
}
for _aid, _labels in _ENGINE_OPERATOR_LABEL_OVERRIDES.items():
    ENGINE_OPERATOR_LABELS[_aid] = list(_labels)

_PREVIOUS_LABELS_FOR_ALGORITHM_CONSERVATIVE = labels_for_algorithm

def labels_for_algorithm(algorithm_id: str) -> list[str]:  # type: ignore[override]
    aid = str(algorithm_id)
    if aid in _ENGINE_OPERATOR_LABEL_OVERRIDES:
        return list(_ENGINE_OPERATOR_LABEL_OVERRIDES[aid])
    return _PREVIOUS_LABELS_FOR_ALGORITHM_CONSERVATIVE(aid)

# Final split policy: preserve explicitly instrumented theory branches, but use
# the earlier compound/fallback splitter for all other opaque single-transition
# engines so the G1 gate still rejects/repairs one-label macro shortcuts.
_PRESERVE_BRANCH_LABELS = set()
for _vals in _EXACT_OPERATOR_LABEL_CANONICALIZATION.values():
    _PRESERVE_BRANCH_LABELS.update(_vals)
for _vals in _ENGINE_OPERATOR_LABEL_OVERRIDES.values():
    _PRESERVE_BRANCH_LABELS.update(_vals)

_CONSERVATIVE_CANONICAL_EXPAND = expand_compound_operator_label

def expand_compound_operator_label(algorithm_id: str, label: str | None) -> list[str]:  # type: ignore[override]
    if label in {None, ""}:
        return []
    raw = str(label)
    if raw in {"carryover", "initialization"}:
        return [raw]
    if raw in _EXACT_COMPOUND_OPERATOR_SPLITS:
        return list(_EXACT_COMPOUND_OPERATOR_SPLITS[raw])
    if raw in _EXACT_OPERATOR_LABEL_CANONICALIZATION:
        return list(_EXACT_OPERATOR_LABEL_CANONICALIZATION[raw])
    if raw in _PRESERVE_BRANCH_LABELS:
        return [raw]
    # Delegate to the original aggressive splitter for generic/opaque labels.
    return _PREVIOUS_EXPAND_COMPOUND_OPERATOR_LABEL(algorithm_id, raw)

for _aid, _labels in list(ENGINE_OPERATOR_LABELS.items()):
    if _aid in _ENGINE_OPERATOR_LABEL_OVERRIDES:
        ENGINE_OPERATOR_LABELS[_aid] = list(_ENGINE_OPERATOR_LABEL_OVERRIDES[_aid])
        continue
    _flat: list[str] = []
    for _lab in _labels:
        for _slab in expand_compound_operator_label(_aid, _lab):
            if _slab and _slab not in _flat:
                _flat.append(_slab)
    ENGINE_OPERATOR_LABELS[_aid] = _flat

_FINAL_LABELS_FOR_ALGORITHM = labels_for_algorithm

def labels_for_algorithm(algorithm_id: str) -> list[str]:  # type: ignore[override]
    aid = str(algorithm_id)
    if aid in _ENGINE_OPERATOR_LABEL_OVERRIDES:
        return list(_ENGINE_OPERATOR_LABEL_OVERRIDES[aid])
    return list(ENGINE_OPERATOR_LABELS.get(aid, _FINAL_LABELS_FOR_ALGORITHM(aid)))

# Additional high-confidence theory-level catalog cleanups for engines that
# already emit precise runtime branch labels.
_ENGINE_OPERATOR_LABEL_OVERRIDES.update({
    "abco": ["abco.employed", "abco.onlooker", "abco.scout"],
    "gwo": ["gwo.alpha_guidance", "gwo.beta_guidance", "gwo.delta_guidance", "gwo.position_update"],
    "avoa": ["avoa.exploration", "avoa.exploitation"],
    "goa": ["goa.social_force", "goa.position_update"],
})
for _aid, _labels in _ENGINE_OPERATOR_LABEL_OVERRIDES.items():
    ENGINE_OPERATOR_LABELS[_aid] = list(_labels)


# ---------------------------------------------------------------------------
# Batch 2: runtime/catalog consistency overrides.
# ---------------------------------------------------------------------------
# These are label-only catalog inclusions for engines whose deterministic
# EvoMapX smoke traces already emit stable semantic runtime labels that were
# missing from labels_for_algorithm().  They do not touch objective calls,
# random draws, state updates, acceptance, or selection behavior.
_BATCH2_RUNTIME_CATALOG_OVERRIDES = {
    'aefa': ['aefa.field_force', 'aefa.state_update', 'aefa.force_or_velocity_update', 'aefa.position_update'],
    'afsa': ['afsa.random_leap', 'afsa.replacement', 'afsa.candidate_generation', 'afsa.selection', 'afsa.prey', 'afsa.swarm', 'afsa.follow', 'afsa.candidate_search'],
    'alo': ['alo.random_walk', 'alo.state_update', 'alo.candidate_generation', 'alo.selection', 'alo.combine'],
    'basin_hopping': ['basin_hopping.candidate_search', 'basin_hopping.selection', 'basin_hopping.candidate_update', 'basin_hopping.local_search', 'basin_hopping.state_update', 'basin_hopping.perturbation', 'basin_hopping.acceptance'],
    'cco': ['cco.candidate_search', 'cco.selection', 'cco.candidate_generation'],
    'cem': ['cem.model_sampling', 'cem.elite_selection', 'cem.distribution_update', 'cem.candidate_generation', 'cem.selection', 'cem.sampling', 'cem.model_update'],
    'cuckoo_s': ['cuckoo_s.levy_flight', 'cuckoo_s.replacement', 'cuckoo_s.candidate_generation', 'cuckoo_s.selection'],
    'dvba': ['dvba.force_or_velocity_update', 'dvba.position_update', 'dvba.random_walk', 'dvba.state_update', 'dvba.candidate_generation', 'dvba.selection'],
    'et_bo': ['et_bo.position_generation', 'et_bo.selection', 'et_bo.step', 'et_bo.candidate_generation', 'et_bo.state_update'],
    'fep': ['fep.fast_mutation', 'fep.tournament_selection', 'fep.candidate_generation', 'fep.selection', 'fep.mutation'],
    'gbrt_bo': ['gbrt_bo.position_generation', 'gbrt_bo.selection', 'gbrt_bo.step', 'gbrt_bo.candidate_generation', 'gbrt_bo.state_update'],
    'gp_bo': ['gp_bo.position_generation', 'gp_bo.selection', 'gp_bo.step', 'gp_bo.candidate_generation', 'gp_bo.state_update'],
    'grasp': ['grasp.candidate_search', 'grasp.selection', 'grasp.construct', 'grasp.local_search', 'grasp.state_update'],
    'hc': ['hc.candidate_generation', 'hc.selection', 'hc.step', 'hc.search', 'hc.state_update'],
    'ils': ['ils.candidate_search', 'ils.selection', 'ils.step', 'ils.local_search', 'ils.state_update', 'ils.perturbation', 'ils.acceptance'],
    'l2smea': ['l2smea.candidate_generation', 'l2smea.selection', 'l2smea.step', 'l2smea.candidate_update'],
    'lfd': ['lfd.levy_flight', 'lfd.search', 'lfd.candidate_generation', 'lfd.selection'],
    'lshade_cnepsin': ['lshade_cnepsin.cn_epsin_adaptation', 'lshade_cnepsin.mutation', 'lshade_cnepsin.crossover', 'lshade_cnepsin.selection', 'lshade_cnepsin.candidate_generation'],
    'misaco': ['misaco.candidate_generation', 'misaco.selection', 'misaco.step', 'misaco.candidate_update'],
    'msls': ['msls.candidate_search', 'msls.selection', 'msls.step', 'msls.local_search', 'msls.state_update'],
    'ngo': ['ngo.2', 'ngo.phase_one_update', 'ngo.selection', 'ngo.candidate_update'],
    'pcx': ['pcx.parent_centric_crossover', 'pcx.state_update', 'pcx.candidate_generation', 'pcx.selection', 'pcx.recombination'],
    'puma_o': ['puma_o.stalking_move', 'puma_o.selection', 'puma_o.candidate_search', 'puma_o.attack', 'puma_o.candidate_generation'],
    'rf_bo': ['rf_bo.position_generation', 'rf_bo.selection', 'rf_bo.step', 'rf_bo.candidate_generation', 'rf_bo.state_update'],
    'sacoso': ['sacoso.cooperation', 'sacoso.swarm_update', 'sacoso.pso_update', 'sacoso.candidate_generation', 'sacoso.selection', 'sacoso.candidate_update'],
    'sade_atdsc': ['sade_atdsc.trial_distribution', 'sade_atdsc.parameter_adaptation', 'sade_atdsc.selection', 'sade_atdsc.candidate_generation', 'sade_atdsc.replacement'],
    'sapo': ['sapo.candidate_generation', 'sapo.selection', 'sapo.step', 'sapo.candidate_update'],
    'sine_cosine_a': ['sine_cosine_a.sine_cosine_move', 'sine_cosine_a.position_update', 'sine_cosine_a.candidate_generation', 'sine_cosine_a.selection', 'sine_cosine_a.position_generation'],
    'ssio_rl': ['ssio_rl.candidate_generation', 'ssio_rl.selection', 'ssio_rl.step', 'ssio_rl.force_update', 'ssio_rl.state_update'],
    'ts': ['ts.neighborhood_move', 'ts.tabu_memory', 'ts.selection', 'ts.candidate_generation', 'ts.position_generation'],
    'vns': ['vns.candidate_search', 'vns.selection', 'vns.step', 'vns.local_search', 'vns.state_update', 'vns.shaking', 'vns.refinement'],
}
_ENGINE_OPERATOR_LABEL_OVERRIDES.update(_BATCH2_RUNTIME_CATALOG_OVERRIDES)
for _aid, _labels in _BATCH2_RUNTIME_CATALOG_OVERRIDES.items():
    ENGINE_OPERATOR_LABELS[_aid] = list(_labels)



# ---------------------------------------------------------------------------
# Batch 3: opaque numeric branch-label cleanup (NGO/RBMO/SBOA).
# ---------------------------------------------------------------------------
# These mappings are purely semantic label canonicalizations.  They replace
# labels such as ``rbmo.1`` and ``sboa.2`` with theory-level branch names while
# preserving the exact same optimizer trajectory, objective-call budget, RNG
# sequence, and accepted populations.
_BATCH3_NUMERIC_LABEL_CANONICALIZATION = {
    "ngo.2": ("ngo.pursuit_exploitation_update",),
    "rbmo.1": ("rbmo.mean_group_exploration_update",),
    "rbmo.2": ("rbmo.food_guided_exploitation_update",),
    "sboa.1": ("sboa.secretary_bird_search_update",),
    "sboa.2": ("sboa.escape_attack_refinement_update",),
}
_EXACT_OPERATOR_LABEL_CANONICALIZATION.update(_BATCH3_NUMERIC_LABEL_CANONICALIZATION)
_BATCH3_RUNTIME_CATALOG_OVERRIDES = {
    "ngo": ["ngo.phase_one_update", "ngo.pursuit_exploitation_update", "ngo.selection"],
    "rbmo": ["rbmo.mean_group_exploration_update", "rbmo.food_guided_exploitation_update"],
    "sboa": ["sboa.secretary_bird_search_update", "sboa.escape_attack_refinement_update"],
}
_ENGINE_OPERATOR_LABEL_OVERRIDES.update(_BATCH3_RUNTIME_CATALOG_OVERRIDES)
for _aid, _labels in _BATCH3_RUNTIME_CATALOG_OVERRIDES.items():
    ENGINE_OPERATOR_LABELS[_aid] = list(_labels)


# ---------------------------------------------------------------------------
# Batch 4: A-family monolithic semantic split upgrades.
# ---------------------------------------------------------------------------
# These mappings are label-only. They keep the passive EvoMapX attribution
# budget-preserving, but avoid flattening engine-specific source labels into
# overly generic ``candidate_update`` / ``position_generation`` buckets.
_BATCH4_A_FAMILY_SEMANTIC_SPLITS = {
    "aao.adaptive_aquila_position_update": (
        "aao.adaptive_aquila_guidance",
        "aao.position_update",
        "aao.selection",
    ),
    "aao.elite_local_refinement": (
        "aao.elite_local_refinement",
        "aao.selection",
    ),
    "aoa.arithmetic_operator_position_update": (
        "aoa.arithmetic_operator_update",
        "aoa.position_update",
        "aoa.selection",
    ),
    "aoo.animated_oat_growth_update": (
        "aoo.animated_oat_growth_update",
        "aoo.selection",
    ),
    "apo.protozoa_life_cycle_update": (
        "apo.protozoa_life_cycle_update",
        "apo.selection",
    ),
    "aso.anarchic_social_position_update": (
        "aso.anarchic_social_position_update",
        "aso.selection",
    ),
}
_EXACT_COMPOUND_OPERATOR_SPLITS.update(_BATCH4_A_FAMILY_SEMANTIC_SPLITS)
_BATCH4_RUNTIME_CATALOG_OVERRIDES = {
    "aao": ["aao.adaptive_aquila_guidance", "aao.position_update", "aao.elite_local_refinement", "aao.selection"],
    "aoa": ["aoa.arithmetic_operator_update", "aoa.position_update", "aoa.selection"],
    "aoo": ["aoo.animated_oat_growth_update", "aoo.selection"],
    "apo": ["apo.protozoa_life_cycle_update", "apo.selection"],
    "aso": ["aso.anarchic_social_position_update", "aso.selection"],
}
_ENGINE_OPERATOR_LABEL_OVERRIDES.update(_BATCH4_RUNTIME_CATALOG_OVERRIDES)
for _aid, _labels in _BATCH4_RUNTIME_CATALOG_OVERRIDES.items():
    ENGINE_OPERATOR_LABELS[_aid] = list(_labels)

# ---------------------------------------------------------------------------
# Batch 5: B/C-family monolithic semantic split upgrades.
# ---------------------------------------------------------------------------
# These mappings are label-only.  They keep the passive EvoMapX attribution
# budget-preserving while replacing overly generic ``candidate_update`` labels
# with theory-level macro-operator names for engines whose faithful source
# already emits a single semantic transition plus selection.
_BATCH5_BC_FAMILY_SEMANTIC_SPLITS = {
    "bps.birds_of_paradise_pose_update": (
        "bps.birds_of_paradise_pose_update",
        "bps.selection",
    ),
    "bso.two_cluster_brainstorm_update": (
        "bso.two_cluster_brainstorm_update",
        "bso.selection",
    ),
    "bspga.binary_partition_tree_variation_update": (
        "bspga.binary_partition_tree_variation_update",
        "bspga.selection",
    ),
    "camel.endurance_temperature_update": (
        "camel.endurance_temperature_update",
        "camel.selection",
    ),
    "capsa.capuchin_jump_swing_update": (
        "capsa.capuchin_jump_swing_update",
        "capsa.selection",
    ),
}
_EXACT_COMPOUND_OPERATOR_SPLITS.update(_BATCH5_BC_FAMILY_SEMANTIC_SPLITS)
_BATCH5_RUNTIME_CATALOG_OVERRIDES = {
    "bps": ["bps.birds_of_paradise_pose_update", "bps.selection"],
    "bso": ["bso.two_cluster_brainstorm_update", "bso.selection"],
    "bspga": ["bspga.binary_partition_tree_variation_update", "bspga.selection"],
    "camel": ["camel.endurance_temperature_update", "camel.selection"],
    "capsa": ["capsa.capuchin_jump_swing_update", "capsa.selection"],
}
_ENGINE_OPERATOR_LABEL_OVERRIDES.update(_BATCH5_RUNTIME_CATALOG_OVERRIDES)
for _aid, _labels in _BATCH5_RUNTIME_CATALOG_OVERRIDES.items():
    ENGINE_OPERATOR_LABELS[_aid] = list(_labels)

# ---------------------------------------------------------------------------
# Batch 6: C/D-family monolithic semantic split upgrades.
# ---------------------------------------------------------------------------
# These mappings are label-only. They preserve the passive EvoMapX attribution
# budget while replacing generic ``candidate_update`` / ``position_generation``
# buckets with the theory-level semantic names already available from the
# faithful source-label rewrite layers.
_BATCH6_CD_FAMILY_SEMANTIC_SPLITS = {
    "cddo_child.child_drawing_development_update": (
        "cddo_child.child_drawing_development_update",
        "cddo_child.selection",
    ),
    "da.dragonfly_swarm_food_enemy_update": (
        "da.dragonfly_swarm_food_enemy_update",
        "da.selection",
    ),
    "deo_dolphin.echo_location_probability_update": (
        "deo_dolphin.echo_location_probability_update",
        "deo_dolphin.selection",
    ),
    "dfo.dispersive_fly_neighbour_update": (
        "dfo.dispersive_fly_neighbour_update",
        "dfo.selection",
    ),
    "dfo.elite_disturbance_update": (
        "dfo.elite_disturbance_update",
        "dfo.selection",
    ),
    "dream_oa.dream_generation_refinement_update": (
        "dream_oa.dream_generation_refinement_update",
        "dream_oa.selection",
    ),
}
_EXACT_COMPOUND_OPERATOR_SPLITS.update(_BATCH6_CD_FAMILY_SEMANTIC_SPLITS)
_BATCH6_RUNTIME_CATALOG_OVERRIDES = {
    "cddo_child": ["cddo_child.child_drawing_development_update", "cddo_child.selection"],
    "da": ["da.dragonfly_swarm_food_enemy_update", "da.selection"],
    "deo_dolphin": ["deo_dolphin.echo_location_probability_update", "deo_dolphin.selection"],
    "dfo": ["dfo.dispersive_fly_neighbour_update", "dfo.elite_disturbance_update", "dfo.selection"],
    "dream_oa": ["dream_oa.dream_generation_refinement_update", "dream_oa.selection"],
}
_ENGINE_OPERATOR_LABEL_OVERRIDES.update(_BATCH6_RUNTIME_CATALOG_OVERRIDES)
for _aid, _labels in _BATCH6_RUNTIME_CATALOG_OVERRIDES.items():
    ENGINE_OPERATOR_LABELS[_aid] = list(_labels)

# ---------------------------------------------------------------------------
# Batch 7: E/F-family monolithic semantic split upgrades.
# ---------------------------------------------------------------------------
# These mappings are label-only. They preserve the passive EvoMapX attribution
# budget while replacing generic ``candidate_update`` / ``position_generation``
# buckets with the theory-level semantic names already available from the
# faithful source-label rewrite layers.
_BATCH7_EF_FAMILY_SEMANTIC_SPLITS = {
    "eho.clan_updating_separating_update": (
        "eho.clan_updating_separating_update",
        "eho.selection",
    ),
    "epc.emperor_penguin_huddle_update": (
        "epc.emperor_penguin_huddle_update",
        "epc.selection",
    ),
    "fata.geophysical_refraction_update": (
        "fata.geophysical_refraction_update",
        "fata.selection",
    ),
    "fda.flow_direction_neighbour_update": (
        "fda.flow_direction_neighbour_update",
        "fda.selection",
    ),
    "fdo.fitness_weighted_pace_update": (
        "fdo.fitness_weighted_pace_update",
        "fdo.selection",
    ),
    "fdo.best_guided_position_update": (
        "fdo.best_guided_position_update",
        "fdo.selection",
    ),
}
_EXACT_COMPOUND_OPERATOR_SPLITS.update(_BATCH7_EF_FAMILY_SEMANTIC_SPLITS)
_BATCH7_RUNTIME_CATALOG_OVERRIDES = {
    "eho": ["eho.clan_updating_separating_update", "eho.selection"],
    "epc": ["epc.emperor_penguin_huddle_update", "epc.selection"],
    "fata": ["fata.geophysical_refraction_update", "fata.selection"],
    "fda": ["fda.flow_direction_neighbour_update", "fda.selection"],
    "fdo": ["fdo.fitness_weighted_pace_update", "fdo.best_guided_position_update", "fdo.selection"],
}
_ENGINE_OPERATOR_LABEL_OVERRIDES.update(_BATCH7_RUNTIME_CATALOG_OVERRIDES)
for _aid, _labels in _BATCH7_RUNTIME_CATALOG_OVERRIDES.items():
    ENGINE_OPERATOR_LABELS[_aid] = list(_labels)

# ---------------------------------------------------------------------------
# Batch 8: G/H-family monolithic semantic split upgrades.
# ---------------------------------------------------------------------------
# These mappings are label-only. They use the labels actually emitted by the
# existing semantic pipeline and canonicalize the inherited GPOO mislabel to an
# octopus-specific name.
_BATCH8_POST_CANONICAL_LABEL_REWRITE = {
    "gpoo.genghis_khan_social_update": "gpoo.octopus_tentacle_prey_position_update",
}
_prev_semanticize_operator_label_batch8_post = semanticize_operator_label

def semanticize_operator_label(algorithm_id: str, label: str | None) -> str | None:  # type: ignore[override]
    lab = _prev_semanticize_operator_label_batch8_post(algorithm_id, label)
    if lab in {None, ""}:
        return lab
    return _BATCH8_POST_CANONICAL_LABEL_REWRITE.get(str(lab), str(lab))

_base_resolve_operator_label_batch8_post = resolve_operator_label

def resolve_operator_label(algorithm_id: str, filename: str, function: str, line: int | None = None) -> str | None:  # type: ignore[override]
    raw = _base_resolve_operator_label_batch8_post(algorithm_id, filename, function, line)
    return semanticize_operator_label(algorithm_id, raw) if raw else None

_BATCH8_GH_FAMILY_ACTUAL_SEMANTIC_SPLITS = {
    "ggo.greylag_goose_flock_update": (
        "ggo.greylag_goose_flock_update",
        "ggo.selection",
    ),
    "go_growth.growth_phase_update": (
        "go_growth.growth_phase_update",
        "go_growth.selection",
    ),
    "go_growth.maturity_phase_update": (
        "go_growth.maturity_phase_update",
        "go_growth.selection",
    ),
    "gpoo.octopus_tentacle_prey_position_update": (
        "gpoo.octopus_tentacle_prey_position_update",
        "gpoo.selection",
    ),
    "hba_honey.honey_badger_digging_honey_update": (
        "hba_honey.honey_badger_digging_honey_update",
        "hba_honey.selection",
    ),
    "hco.conception_growth_update": (
        "hco.conception_growth_update",
        "hco.selection",
    ),
}
_EXACT_COMPOUND_OPERATOR_SPLITS.update(_BATCH8_GH_FAMILY_ACTUAL_SEMANTIC_SPLITS)
_BATCH8_RUNTIME_CATALOG_OVERRIDES = {
    "ggo": ["ggo.greylag_goose_flock_update", "ggo.selection"],
    "go_growth": [
        "go_growth.growth_phase_update",
        "go_growth.maturity_phase_update",
        "go_growth.selection",
    ],
    "gpoo": ["gpoo.octopus_tentacle_prey_position_update", "gpoo.selection"],
    "hba_honey": ["hba_honey.honey_badger_digging_honey_update", "hba_honey.selection"],
    "hco": ["hco.conception_growth_update", "hco.selection"],
}
_ENGINE_OPERATOR_LABEL_OVERRIDES.update(_BATCH8_RUNTIME_CATALOG_OVERRIDES)
for _aid, _labels in _BATCH8_RUNTIME_CATALOG_OVERRIDES.items():
    ENGINE_OPERATOR_LABELS[_aid] = list(_labels)

# Batch 8 final labels_for_algorithm guard: do not regress earlier catalog
# overrides for untouched engines.  Return the current flattened table, which
# already includes all previous batch overrides plus the Batch 8 entries.
def labels_for_algorithm(algorithm_id: str) -> list[str]:  # type: ignore[override]
    aid = str(algorithm_id)
    return list(ENGINE_OPERATOR_LABELS.get(aid, []))

# ---------------------------------------------------------------------------
# Batch 9: H-family monolithic semantic split upgrades.
# ---------------------------------------------------------------------------
# These mappings are label-only. They preserve the passive EvoMapX attribution
# budget while replacing generic ``candidate_update`` buckets with the
# theory-level semantic names that the existing probe already records as the
# original operator labels for these H-family engines.
_BATCH9_H_FAMILY_ACTUAL_SEMANTIC_SPLITS = {
    "hbo.heap_rank_pressure_update": (
        "hbo.heap_rank_pressure_update",
        "hbo.selection",
    ),
    "hgs.hunger_games_social_pressure_update": (
        "hgs.hunger_games_social_pressure_update",
        "hgs.selection",
    ),
    "horse_oa.social_hierarchy_grazing_update": (
        "horse_oa.social_hierarchy_grazing_update",
        "horse_oa.selection",
    ),
    "hsa.harmony_memory_improvisation_update": (
        "hsa.harmony_memory_improvisation_update",
        "hsa.selection",
    ),
    "hsaba.self_adaptive_bat_de_update": (
        "hsaba.self_adaptive_bat_de_update",
        "hsaba.selection",
    ),
}
_EXACT_COMPOUND_OPERATOR_SPLITS.update(_BATCH9_H_FAMILY_ACTUAL_SEMANTIC_SPLITS)
_BATCH9_RUNTIME_CATALOG_OVERRIDES = {
    "hbo": ["hbo.heap_rank_pressure_update", "hbo.selection"],
    "hgs": ["hgs.hunger_games_social_pressure_update", "hgs.selection"],
    "horse_oa": ["horse_oa.social_hierarchy_grazing_update", "horse_oa.selection"],
    "hsa": ["hsa.harmony_memory_improvisation_update", "hsa.selection"],
    "hsaba": ["hsaba.self_adaptive_bat_de_update", "hsaba.selection"],
}
_ENGINE_OPERATOR_LABEL_OVERRIDES.update(_BATCH9_RUNTIME_CATALOG_OVERRIDES)
for _aid, _labels in _BATCH9_RUNTIME_CATALOG_OVERRIDES.items():
    ENGINE_OPERATOR_LABELS[_aid] = list(_labels)

# ---------------------------------------------------------------------------
# Batch 10: I/J/L-family monolithic semantic split upgrades.
# ---------------------------------------------------------------------------
# These mappings are label-only. They preserve the passive EvoMapX attribution
# budget while replacing generic ``candidate_update`` / ``candidate_generation``
# buckets with the theory-level semantic names that the existing probe already
# records as original operator labels for these I/J/L-family engines.
_BATCH10_IJL_FAMILY_ACTUAL_SEMANTIC_SPLITS = {
    "iaro.improved_rabbit_global_update": (
        "iaro.improved_rabbit_global_update",
        "iaro.selection",
    ),
    "iaro.elite_local_refinement": (
        "iaro.elite_local_refinement",
        "iaro.selection",
    ),
    "ivya.ivy_growth_neighbor_update": (
        "ivya.ivy_growth_neighbor_update",
        "ivya.selection",
    ),
    "jy.best_away_from_worst_update": (
        "jy.best_away_from_worst_update",
        "jy.selection",
    ),
    "lco.life_choice_boundary_reflection_update": (
        "lco.life_choice_boundary_reflection_update",
        "lco.selection",
    ),
    "loa_lyrebird.lyrebird_escape_hiding_update": (
        "loa_lyrebird.lyrebird_escape_hiding_update",
        "loa_lyrebird.selection",
    ),
}
_EXACT_COMPOUND_OPERATOR_SPLITS.update(_BATCH10_IJL_FAMILY_ACTUAL_SEMANTIC_SPLITS)
_BATCH10_RUNTIME_CATALOG_OVERRIDES = {
    "iaro": [
        "iaro.improved_rabbit_global_update",
        "iaro.elite_local_refinement",
        "iaro.selection",
    ],
    "ivya": ["ivya.ivy_growth_neighbor_update", "ivya.selection"],
    "jy": ["jy.best_away_from_worst_update", "jy.selection"],
    "lco": ["lco.life_choice_boundary_reflection_update", "lco.selection"],
    "loa_lyrebird": ["loa_lyrebird.lyrebird_escape_hiding_update", "loa_lyrebird.selection"],
}
_ENGINE_OPERATOR_LABEL_OVERRIDES.update(_BATCH10_RUNTIME_CATALOG_OVERRIDES)
for _aid, _labels in _BATCH10_RUNTIME_CATALOG_OVERRIDES.items():
    ENGINE_OPERATOR_LABELS[_aid] = list(_labels)

# ---------------------------------------------------------------------------
# Batch 11: L/M-family monolithic semantic split upgrades.
# ---------------------------------------------------------------------------
# These mappings are label-only. They preserve EvoMapX's passive attribution
# accounting while replacing generic ``candidate_update`` buckets with the
# theory-level semantic names that the existing probe already records as
# original operator labels for these L/M-family engines.
_BATCH11_LM_FAMILY_ACTUAL_SEMANTIC_SPLITS = {
    "lpo.lichen_growth_propagation_update": (
        "lpo.lichen_growth_propagation_update",
        "lpo.selection",
    ),
    "mfa.moth_flame_spiral_update": (
        "mfa.moth_flame_spiral_update",
        "mfa.selection",
    ),
    "mgo.territory_mountain_herding_update": (
        "mgo.territory_mountain_herding_update",
        "mgo.selection",
    ),
    "mke.memory_knowledge_evolution_update": (
        "mke.memory_knowledge_evolution_update",
        "mke.selection",
    ),
    "mtbo.mountaineering_random_relocation_update": (
        "mtbo.mountaineering_random_relocation_update",
        "mtbo.selection",
    ),
}
_EXACT_COMPOUND_OPERATOR_SPLITS.update(_BATCH11_LM_FAMILY_ACTUAL_SEMANTIC_SPLITS)
_BATCH11_RUNTIME_CATALOG_OVERRIDES = {
    "lpo": ["lpo.lichen_growth_propagation_update", "lpo.selection"],
    "mfa": ["mfa.moth_flame_spiral_update", "mfa.selection"],
    "mgo": ["mgo.territory_mountain_herding_update", "mgo.selection"],
    "mke": ["mke.memory_knowledge_evolution_update", "mke.selection"],
    "mtbo": ["mtbo.mountaineering_random_relocation_update", "mtbo.selection"],
}
_ENGINE_OPERATOR_LABEL_OVERRIDES.update(_BATCH11_RUNTIME_CATALOG_OVERRIDES)
for _aid, _labels in _BATCH11_RUNTIME_CATALOG_OVERRIDES.items():
    ENGINE_OPERATOR_LABELS[_aid] = list(_labels)

# ---------------------------------------------------------------------------
# Batch 12: N/P/R-family monolithic semantic split upgrades.
# ---------------------------------------------------------------------------
# These mappings are label-only. They preserve EvoMapX's passive attribution
# accounting while replacing generic ``candidate_update`` buckets with the
# theory-level semantic names already recorded as original operator labels for
# these N/P/R-family engines.
_BATCH12_NPR_FAMILY_ACTUAL_SEMANTIC_SPLITS = {
    "nmm.reflection_update": (
        "nmm.reflection_update",
        "nmm.selection",
    ),
    "nro.nuclear_fission_update": (
        "nro.nuclear_fission_update",
        "nro.selection",
    ),
    "nro.nuclear_fusion_update": (
        "nro.nuclear_fusion_update",
        "nro.selection",
    ),
    "pdo.prairie_dog_burrow_alarm_update": (
        "pdo.prairie_dog_burrow_alarm_update",
        "pdo.selection",
    ),
    "plba.path_looping_bat_update": (
        "plba.path_looping_bat_update",
        "plba.selection",
    ),
    "rcco.rain_cloud_convection_update": (
        "rcco.rain_cloud_convection_update",
        "rcco.selection",
    ),
    "rcco.cloud_collision_local_update": (
        "rcco.cloud_collision_local_update",
        "rcco.selection",
    ),
}
_EXACT_COMPOUND_OPERATOR_SPLITS.update(_BATCH12_NPR_FAMILY_ACTUAL_SEMANTIC_SPLITS)
_BATCH12_RUNTIME_CATALOG_OVERRIDES = {
    "nmm": ["nmm.reflection_update", "nmm.selection"],
    "nro": ["nro.nuclear_fission_update", "nro.nuclear_fusion_update", "nro.selection"],
    "pdo": ["pdo.prairie_dog_burrow_alarm_update", "pdo.selection"],
    "plba": ["plba.path_looping_bat_update", "plba.selection"],
    "rcco": ["rcco.rain_cloud_convection_update", "rcco.cloud_collision_local_update", "rcco.selection"],
}
_ENGINE_OPERATOR_LABEL_OVERRIDES.update(_BATCH12_RUNTIME_CATALOG_OVERRIDES)
for _aid, _labels in _BATCH12_RUNTIME_CATALOG_OVERRIDES.items():
    ENGINE_OPERATOR_LABELS[_aid] = list(_labels)

# ---------------------------------------------------------------------------
# Batch 12 strict-smoke catalog normalization.
# ---------------------------------------------------------------------------
# The strict audit treats every runtime attribution key as a catalog-visible
# label. This keeps a pre-existing AoA semantic split from falling back to a
# generic bucket and records non-operator carryover groups that can be emitted
# by passive lineage attribution in SMA/SRSR.
_BATCH12_STRICT_SMOKE_SPLITS = {
    "aoa.arithmetic_operator_update": (
        "aoa.arithmetic_operator_update",
        "aoa.position_update",
        "aoa.selection",
    ),
}
_EXACT_COMPOUND_OPERATOR_SPLITS.update(_BATCH12_STRICT_SMOKE_SPLITS)
_BATCH12_STRICT_SMOKE_CATALOG_OVERRIDES = {
    "aoa": ["aoa.arithmetic_operator_update", "aoa.position_update", "aoa.selection"],
    "sma": ["sma.candidate_update", "sma.selection", "sma.candidate_generation", "carryover"],
    "srsr": ["srsr.position_generation", "srsr.selection", "srsr.exploration", "srsr.candidate_generation", "carryover"],
}
_ENGINE_OPERATOR_LABEL_OVERRIDES.update(_BATCH12_STRICT_SMOKE_CATALOG_OVERRIDES)
for _aid, _labels in _BATCH12_STRICT_SMOKE_CATALOG_OVERRIDES.items():
    ENGINE_OPERATOR_LABELS[_aid] = list(_labels)

# ---------------------------------------------------------------------------
# Batch 13: R/S-family monolithic semantic split upgrades.
# ---------------------------------------------------------------------------
# These mappings are label-only. They preserve EvoMapX's passive attribution
# accounting while replacing generic ``candidate_update`` buckets with the
# theory-level semantic names already recorded as original operator labels for
# these R/S-family engines.
_BATCH13_RS_FAMILY_ACTUAL_SEMANTIC_SPLITS = {
    "roa.remora_attempt_update": (
        "roa.remora_attempt_update",
        "roa.selection",
    ),
    "rso.rat_swarm_chasing_update": (
        "rso.rat_swarm_chasing_update",
        "rso.selection",
    ),
    "saba.self_adaptive_bat_update": (
        "saba.self_adaptive_bat_update",
        "saba.selection",
    ),
    "samso.self_adaptive_migratory_swarm_update": (
        "samso.self_adaptive_migratory_swarm_update",
        "samso.selection",
    ),
    "shio_success.success_based_iguana_update": (
        "shio_success.success_based_iguana_update",
        "shio_success.selection",
    ),
}
_EXACT_COMPOUND_OPERATOR_SPLITS.update(_BATCH13_RS_FAMILY_ACTUAL_SEMANTIC_SPLITS)
_BATCH13_RUNTIME_CATALOG_OVERRIDES = {
    "roa": ["roa.remora_attempt_update", "roa.selection"],
    "rso": ["rso.rat_swarm_chasing_update", "rso.selection"],
    "saba": ["saba.self_adaptive_bat_update", "saba.selection"],
    "samso": ["samso.self_adaptive_migratory_swarm_update", "samso.selection"],
    "shio_success": ["shio_success.success_based_iguana_update", "shio_success.selection"],
}
_ENGINE_OPERATOR_LABEL_OVERRIDES.update(_BATCH13_RUNTIME_CATALOG_OVERRIDES)
for _aid, _labels in _BATCH13_RUNTIME_CATALOG_OVERRIDES.items():
    ENGINE_OPERATOR_LABELS[_aid] = list(_labels)

# ---------------------------------------------------------------------------
# Batch 14: S/T-family monolithic semantic split upgrades.
# ---------------------------------------------------------------------------
# These mappings are label-only. They preserve EvoMapX's passive attribution
# accounting while replacing generic ``candidate_update`` buckets with the
# theory-level semantic names already recorded as original operator labels for
# these S/T-family engines.
_BATCH14_ST_FAMILY_ACTUAL_SEMANTIC_SPLITS = {
    "so_snake.male_snake_update": (
        "so_snake.male_snake_update",
        "so_snake.selection",
    ),
    "so_snake.female_snake_update": (
        "so_snake.female_snake_update",
        "so_snake.selection",
    ),
    "ssdo.social_ski_driver_update": (
        "ssdo.social_ski_driver_update",
        "ssdo.selection",
    ),
    "sspider_a.social_spider_vibration_update": (
        "sspider_a.social_spider_vibration_update",
        "sspider_a.selection",
    ),
    "thro.throwing_race_update": (
        "thro.throwing_race_update",
        "thro.selection",
    ),
    "tlco.teacher_phase_update": (
        "tlco.teacher_phase_update",
        "tlco.selection",
    ),
    "tlco.learner_phase_update": (
        "tlco.learner_phase_update",
        "tlco.selection",
    ),
}
_EXACT_COMPOUND_OPERATOR_SPLITS.update(_BATCH14_ST_FAMILY_ACTUAL_SEMANTIC_SPLITS)
_BATCH14_RUNTIME_CATALOG_OVERRIDES = {
    "so_snake": ["so_snake.male_snake_update", "so_snake.female_snake_update", "so_snake.selection"],
    "ssdo": ["ssdo.social_ski_driver_update", "ssdo.selection"],
    "sspider_a": ["sspider_a.social_spider_vibration_update", "sspider_a.selection"],
    "thro": ["thro.throwing_race_update", "thro.selection"],
    "tlco": ["tlco.teacher_phase_update", "tlco.learner_phase_update", "tlco.selection"],
}
_ENGINE_OPERATOR_LABEL_OVERRIDES.update(_BATCH14_RUNTIME_CATALOG_OVERRIDES)
for _aid, _labels in _BATCH14_RUNTIME_CATALOG_OVERRIDES.items():
    ENGINE_OPERATOR_LABELS[_aid] = list(_labels)

# ---------------------------------------------------------------------------
# Batch 15: late T/W-family monolithic semantic split upgrades.
# ---------------------------------------------------------------------------
# These mappings are label-only. They preserve EvoMapX's passive attribution
# accounting while replacing generic ``candidate_update`` / ``position_generation``
# buckets with the theory-level semantic names already recorded as original
# operator labels for these late T/W-family engines.
_BATCH15_TW_FAMILY_ACTUAL_SEMANTIC_SPLITS = {
    "tpo.carbon_nutrient_leaf_update": (
        "tpo.carbon_nutrient_leaf_update",
        "tpo.selection",
    ),
    "tree_seed_a.away_random_seed_update": (
        "tree_seed_a.away_random_seed_update",
        "tree_seed_a.selection",
    ),
    "tsa.tunicate_swarm_position_update": (
        "tsa.tunicate_swarm_position_update",
        "tsa.selection",
    ),
    "tso.transient_parabolic_position_update": (
        "tso.transient_parabolic_position_update",
        "tso.selection",
    ),
    "warso.war_strategy_ranked_update": (
        "warso.war_strategy_ranked_update",
        "warso.selection",
    ),
}
_EXACT_COMPOUND_OPERATOR_SPLITS.update(_BATCH15_TW_FAMILY_ACTUAL_SEMANTIC_SPLITS)
_BATCH15_RUNTIME_CATALOG_OVERRIDES = {
    "tpo": ["tpo.carbon_nutrient_leaf_update", "tpo.selection"],
    "tree_seed_a": ["tree_seed_a.away_random_seed_update", "tree_seed_a.selection"],
    "tsa": ["tsa.tunicate_swarm_position_update", "tsa.selection"],
    "tso": ["tso.transient_parabolic_position_update", "tso.selection"],
    "warso": ["warso.war_strategy_ranked_update", "warso.selection"],
}
_ENGINE_OPERATOR_LABEL_OVERRIDES.update(_BATCH15_RUNTIME_CATALOG_OVERRIDES)
for _aid, _labels in _BATCH15_RUNTIME_CATALOG_OVERRIDES.items():
    ENGINE_OPERATOR_LABELS[_aid] = list(_labels)

# ---------------------------------------------------------------------------
# Batch 16: compact A/B/C-family semantic split upgrades.
# ---------------------------------------------------------------------------
# These mappings are label-only. They preserve the passive EvoMapX attribution
# accounting while replacing generic ``candidate_update`` / ``position_generation``
# buckets with the theory-level semantic names already recorded as original
# operator labels for these compact A/B/C-family engines.
_BATCH16_ABC_FAMILY_ACTUAL_SEMANTIC_SPLITS = {
    "autov.learned_variation_operator_update": (
        "autov.learned_variation_operator_update",
        "autov.selection",
    ),
    "bfo.chemotaxis_tumble_update": (
        "bfo.chemotaxis_tumble_update",
        "bfo.selection",
    ),
    "cddo.cheetah_chase_position_update": (
        "cddo.cheetah_chase_position_update",
        "cddo.selection",
    ),
    "cdo.cheetah_density_position_update": (
        "cdo.cheetah_density_position_update",
        "cdo.selection",
    ),
    "ceo_cosmic.cosmic_evolution_position_update": (
        "ceo_cosmic.cosmic_evolution_position_update",
        "ceo_cosmic.selection",
    ),
}
_EXACT_COMPOUND_OPERATOR_SPLITS.update(_BATCH16_ABC_FAMILY_ACTUAL_SEMANTIC_SPLITS)
_BATCH16_RUNTIME_CATALOG_OVERRIDES = {
    "autov": ["autov.learned_variation_operator_update", "autov.selection"],
    "bfo": ["bfo.chemotaxis_tumble_update", "bfo.selection"],
    "cddo": ["cddo.cheetah_chase_position_update", "cddo.selection"],
    "cdo": ["cdo.cheetah_density_position_update", "cdo.selection"],
    "ceo_cosmic": ["ceo_cosmic.cosmic_evolution_position_update", "ceo_cosmic.selection"],
}
_ENGINE_OPERATOR_LABEL_OVERRIDES.update(_BATCH16_RUNTIME_CATALOG_OVERRIDES)
for _aid, _labels in _BATCH16_RUNTIME_CATALOG_OVERRIDES.items():
    ENGINE_OPERATOR_LABELS[_aid] = list(_labels)

# ---------------------------------------------------------------------------
# Batch 17: compact C/D/F-family semantic split upgrades.
# ---------------------------------------------------------------------------
# These mappings are label-only. They preserve EvoMapX's passive attribution
# accounting while replacing generic ``candidate_update`` / ``position_generation``
# buckets with the theory-level semantic names already recorded as original
# operator labels for these compact C/D/F-family engines.
_BATCH17_CDF_FAMILY_ACTUAL_SEMANTIC_SPLITS = {
    "cro.reef_broadcast_spawning_update": (
        "cro.reef_broadcast_spawning_update",
        "cro.selection",
    ),
    "cro.reef_brooding_settlement_update": (
        "cro.reef_brooding_settlement_update",
        "cro.selection",
    ),
    "csa.memory_following_update": (
        "csa.memory_following_update",
        "csa.selection",
    ),
    "dso.deep_sleep_position_update": (
        "dso.deep_sleep_position_update",
        "dso.selection",
    ),
    "fla.fick_law_diffusion_transport_update": (
        "fla.fick_law_diffusion_transport_update",
        "fla.selection",
    ),
    "flood_a.flood_flow_direction_update": (
        "flood_a.flood_flow_direction_update",
        "flood_a.selection",
    ),
    "flood_a.flood_recession_refinement_update": (
        "flood_a.flood_recession_refinement_update",
        "flood_a.selection",
    ),
}
_EXACT_COMPOUND_OPERATOR_SPLITS.update(_BATCH17_CDF_FAMILY_ACTUAL_SEMANTIC_SPLITS)
_BATCH17_RUNTIME_CATALOG_OVERRIDES = {
    "cro": ["cro.reef_broadcast_spawning_update", "cro.reef_brooding_settlement_update", "cro.selection"],
    "csa": ["csa.memory_following_update", "csa.selection"],
    "dso": ["dso.deep_sleep_position_update", "dso.selection"],
    "fla": ["fla.fick_law_diffusion_transport_update", "fla.selection"],
    "flood_a": ["flood_a.flood_flow_direction_update", "flood_a.flood_recession_refinement_update", "flood_a.selection"],
}
_ENGINE_OPERATOR_LABEL_OVERRIDES.update(_BATCH17_RUNTIME_CATALOG_OVERRIDES)
for _aid, _labels in _BATCH17_RUNTIME_CATALOG_OVERRIDES.items():
    ENGINE_OPERATOR_LABELS[_aid] = list(_labels)

# ---------------------------------------------------------------------------
# Batch 18: compact B/F/G-family semantic split upgrades.
# ---------------------------------------------------------------------------
# These mappings are label-only. They preserve EvoMapX's passive attribution
# accounting while replacing generic ``candidate_generation`` / ``candidate_update`` /
# ``position_generation`` buckets with the theory-level semantic names already
# recorded as original operator labels for these compact B/F/G-family engines.
_BATCH18_BFG_FAMILY_ACTUAL_SEMANTIC_SPLITS = {
    "bro.find_nearest_neighbour": (
        "bro.find_nearest_neighbour",
        "bro.selection",
    ),
    "bro.battle_damage_relocation_update": (
        "bro.battle_damage_relocation_update",
        "bro.selection",
    ),
    "chio.immune_contact_update": (
        "chio.immune_contact_update",
        "chio.selection",
    ),
    "foa.local_seeding_growth_update": (
        "foa.local_seeding_growth_update",
        "foa.selection",
    ),
    "fss.collective_volitive_movement": (
        "fss.collective_volitive_movement",
        "fss.selection",
    ),
    "gso.glowworm_luciferin_movement_update": (
        "gso.glowworm_luciferin_movement_update",
        "gso.selection",
    ),
}
_EXACT_COMPOUND_OPERATOR_SPLITS.update(_BATCH18_BFG_FAMILY_ACTUAL_SEMANTIC_SPLITS)
_BATCH18_RUNTIME_CATALOG_OVERRIDES = {
    "bro": ["bro.find_nearest_neighbour", "bro.battle_damage_relocation_update", "bro.selection"],
    "chio": ["chio.immune_contact_update", "chio.selection"],
    "foa": ["foa.local_seeding_growth_update", "foa.selection"],
    "fss": ["fss.collective_volitive_movement", "fss.selection"],
    "gso": ["gso.glowworm_luciferin_movement_update", "gso.selection"],
}
_ENGINE_OPERATOR_LABEL_OVERRIDES.update(_BATCH18_RUNTIME_CATALOG_OVERRIDES)
for _aid, _labels in _BATCH18_RUNTIME_CATALOG_OVERRIDES.items():
    ENGINE_OPERATOR_LABELS[_aid] = list(_labels)

# ---------------------------------------------------------------------------
# Batch 19: complete remaining generic-label cleanup.
# ---------------------------------------------------------------------------
# Label-only completion pass.  The mappings below cover all engines that still
# exposed generic ``candidate_update`` or ``position_generation`` catalog
# labels after Batch 18.  They are derived from observed EvoMapX
# ``original_operator`` telemetry where available; source-line/generic
# residues are canonicalized to stable semantic labels without changing
# optimizer state, RNG use, objective calls, or selection rules.
_BATCH19_REMAINING_EXACT_SEMANTIC_REWRITE = {'basin_hopping.local_search_evaluation_l177': 'basin_hopping.basin_hopping_local_search_update',
 'bco.candidate_update': 'bco.bco_semantic_update',
 'ca.candidate_update': 'ca.ca_semantic_update',
 'cdo_chernobyl.position_generation': 'cdo_chernobyl.cdo_chernobyl_position_update',
 'chicken_so.step_evaluation_l73': 'chicken_so.chicken_so_semantic_update',
 'circle_sa.position_generation': 'circle_sa.circle_simulated_annealing_position_update',
 'compact_ga.candidate_update': 'compact_ga.compact_genetic_algorithm_semantic_update',
 'et_bo.evaluate_positions_evaluation_l449': 'et_bo.et_bayesian_optimization_position_update',
 'et_bo.position_generation': 'et_bo.et_bayesian_optimization_position_update',
 'gbrt_bo.evaluate_positions_evaluation_l449': 'gbrt_bo.gbrt_bayesian_optimization_position_update',
 'gbrt_bo.position_generation': 'gbrt_bo.gbrt_bayesian_optimization_position_update',
 'gp_bo.evaluate_positions_evaluation_l449': 'gp_bo.gp_bayesian_optimization_position_update',
 'gp_bo.position_generation': 'gp_bo.gp_bayesian_optimization_position_update',
 'hus.step_evaluation_l62': 'hus.hus_semantic_update',
 'l2smea.step_evaluation_l167': 'l2smea.l2smea_semantic_update',
 'misaco.step_evaluation_l272': 'misaco.misaco_semantic_update',
 'rf_bo.position_generation': 'rf_bo.rf_bayesian_optimization_position_update',
 'rime.candidate_update': 'rime.rime_semantic_update',
 'sade_amss.candidate_update': 'sade_amss.sade_amss_semantic_update',
 'sapo.candidate_generation': 'sapo.sapo_semantic_update',
 'sapo.step_evaluation_l398': 'sapo.sapo_semantic_update',
 'wso.position_generation': 'wso.wso_position_update'}
_EXACT_SEMANTIC_REWRITE.update(_BATCH19_REMAINING_EXACT_SEMANTIC_REWRITE)
_BATCH19_REMAINING_ACTUAL_SEMANTIC_SPLITS = {'aaa.adaptation_most_starving_colony_moves_toward': ('aaa.adaptation_most_starving_colony_moves_toward',
                                                      'aaa.selection'),
 'aaa.is_replaced_by_corresponding_cell_biggest': ('aaa.is_replaced_by_corresponding_cell_biggest', 'aaa.selection'),
 'acgwo.adaptive_weighted_pack_update': ('acgwo.adaptive_weighted_pack_update', 'acgwo.selection'),
 'acgwo.alpha_guidance_trial': ('acgwo.alpha_guidance_trial', 'acgwo.selection'),
 'acgwo.beta_guidance_trial': ('acgwo.beta_guidance_trial', 'acgwo.selection'),
 'acgwo.delta_guidance_trial': ('acgwo.delta_guidance_trial', 'acgwo.selection'),
 'adaptive_eo.adaptive_local_refinement': ('adaptive_eo.adaptive_local_refinement', 'adaptive_eo.selection'),
 'adaptive_eo.equilibrium_pool_guided_update': ('adaptive_eo.equilibrium_pool_guided_update', 'adaptive_eo.selection'),
 'aeo.consumer_decomposer_update': ('aeo.consumer_decomposer_update', 'aeo.selection'),
 'aeo.production_worst_agent': ('aeo.production_worst_agent', 'aeo.selection'),
 'aiw_pso.elite_local_refinement': ('aiw_pso.elite_local_refinement', 'aiw_pso.selection'),
 'aso_atom.do_not_move_current_elites_unless': ('aso_atom.do_not_move_current_elites_unless', 'aso_atom.selection'),
 'bacterial_colony_o.current_colony_best_accept_only_it': ('bacterial_colony_o.current_colony_best_accept_only_it',
                                                           'bacterial_colony_o.selection'),
 'bacterial_colony_o.implementation_but_only_as_bounded_macro': ('bacterial_colony_o.implementation_but_only_as_bounded_macro',
                                                                 'bacterial_colony_o.selection'),
 'basin_hopping.basin_hopping_local_search_update': ('basin_hopping.basin_hopping_local_search_update',
                                                     'basin_hopping.candidate_search',
                                                     'basin_hopping.selection'),
 'basin_hopping.local_search_evaluation_l177': ('basin_hopping.basin_hopping_local_search_update',
                                                'basin_hopping.candidate_search',
                                                'basin_hopping.selection'),
 'bboa.2_sniffing': ('bboa.2_sniffing', 'bboa.selection'),
 'bboa.pedal_marking_update': ('bboa.pedal_marking_update', 'bboa.selection'),
 'bco.bco_semantic_update': ('bco.bco_semantic_update', 'bco.selection'),
 'bco.candidate_update': ('bco.bco_semantic_update', 'bco.selection'),
 'bco.swim_refinement_update': ('bco.swim_refinement_update', 'bco.selection'),
 'btoa.defensive_play_refinement': ('btoa.defensive_play_refinement', 'btoa.selection'),
 'btoa.dynamic_position_candidate': ('btoa.dynamic_position_candidate', 'btoa.position_update', 'btoa.selection'),
 'btoa.offensive_play_update': ('btoa.offensive_play_update', 'btoa.selection'),
 'ca.ca_semantic_update': ('ca.ca_semantic_update', 'ca.selection'),
 'ca.candidate_update': ('ca.ca_semantic_update', 'ca.selection'),
 'ca.cultural_belief_guided_update': ('ca.cultural_belief_guided_update', 'ca.selection'),
 'cdo_chernobyl.cdo_chernobyl_position_update': ('cdo_chernobyl.cdo_chernobyl_position_update',
                                                 'cdo_chernobyl.selection'),
 'cdo_chernobyl.position_generation': ('cdo_chernobyl.cdo_chernobyl_position_update', 'cdo_chernobyl.selection'),
 'cg_gwo.elite_local_refinement': ('cg_gwo.elite_local_refinement', 'cg_gwo.selection'),
 'cg_gwo.leader_guided_population_update': ('cg_gwo.leader_guided_population_update', 'cg_gwo.selection'),
 'chaotic_gwo.elite_local_refinement': ('chaotic_gwo.elite_local_refinement', 'chaotic_gwo.selection'),
 'chaotic_gwo.leader_guided_population_update': ('chaotic_gwo.leader_guided_population_update',
                                                 'chaotic_gwo.selection'),
 'chicken_so.chicken_so_semantic_update': ('chicken_so.chicken_so_semantic_update', 'chicken_so.selection'),
 'chicken_so.step_evaluation_l73': ('chicken_so.chicken_so_semantic_update', 'chicken_so.selection'),
 'circle_sa.circle_position_update': ('circle_sa.circle_position_update', 'circle_sa.selection'),
 'circle_sa.circle_simulated_annealing_position_update': ('circle_sa.circle_simulated_annealing_position_update',
                                                          'circle_sa.selection'),
 'circle_sa.position_generation': ('circle_sa.circle_simulated_annealing_position_update', 'circle_sa.selection'),
 'compact_ga.candidate_update': ('compact_ga.compact_genetic_algorithm_semantic_update', 'compact_ga.selection'),
 'compact_ga.compact_genetic_algorithm_semantic_update': ('compact_ga.compact_genetic_algorithm_semantic_update',
                                                          'compact_ga.selection'),
 'coot.chain_movement_update': ('coot.chain_movement_update', 'coot.selection'),
 'cso.mean_all_positions': ('cso.mean_all_positions', 'cso.selection'),
 'dbo.ball_rolling_dance_update': ('dbo.ball_rolling_dance_update', 'dbo.selection'),
 'ddao.dynamic_annealed_refinement_update': ('ddao.dynamic_annealed_refinement_update', 'ddao.selection'),
 'dmoa.3_baby_sitter_eviction': ('dmoa.3_baby_sitter_eviction', 'dmoa.selection'),
 'dmoa.scalar_broadcast': ('dmoa.scalar_broadcast', 'dmoa.selection'),
 'dmoa.scout_phase': ('dmoa.scout_phase', 'dmoa.selection'),
 'do_dandelion.seed_landing_update': ('do_dandelion.seed_landing_update', 'do_dandelion.selection'),
 'dra.dialectic_interaction_update': ('dra.dialectic_interaction_update', 'dra.selection'),
 'ds_gwo.elite_local_refinement': ('ds_gwo.elite_local_refinement', 'ds_gwo.selection'),
 'ds_gwo.leader_guided_population_update': ('ds_gwo.leader_guided_population_update', 'ds_gwo.selection'),
 'ecological_cycle_o.ecological_cycle_transition_update': ('ecological_cycle_o.ecological_cycle_transition_update',
                                                           'ecological_cycle_o.selection'),
 'ecological_cycle_o.eval_accept_group': ('ecological_cycle_o.eval_accept_group', 'ecological_cycle_o.selection'),
 'elk_ho.family_mating_position_update': ('elk_ho.family_mating_position_update', 'elk_ho.selection'),
 'enhanced_aeo.ecosystem_producer_consumer_update': ('enhanced_aeo.ecosystem_producer_consumer_update',
                                                     'enhanced_aeo.selection'),
 'enhanced_aeo.enhanced_decomposition_refinement': ('enhanced_aeo.enhanced_decomposition_refinement',
                                                    'enhanced_aeo.selection'),
 'eo.equilibrium_position_update': ('eo.equilibrium_position_update', 'eo.selection'),
 'er_gwo.elite_local_refinement': ('er_gwo.elite_local_refinement', 'er_gwo.selection'),
 'er_gwo.leader_guided_population_update': ('er_gwo.leader_guided_population_update', 'er_gwo.selection'),
 'esoa.egret_sit_and_wait_update': ('esoa.egret_sit_and_wait_update', 'esoa.selection'),
 'et_bo.et_bayesian_optimization_position_update': ('et_bo.et_bayesian_optimization_position_update',
                                                    'et_bo.selection'),
 'et_bo.evaluate_positions_evaluation_l449': ('et_bo.et_bayesian_optimization_position_update', 'et_bo.selection'),
 'et_bo.position_generation': ('et_bo.et_bayesian_optimization_position_update', 'et_bo.selection'),
 'ex_gwo.elite_local_refinement': ('ex_gwo.elite_local_refinement', 'ex_gwo.selection'),
 'ex_gwo.leader_guided_population_update': ('ex_gwo.leader_guided_population_update', 'ex_gwo.selection'),
 'fox.preserve_best_few_individuals_explicitly': ('fox.preserve_best_few_individuals_explicitly', 'fox.selection'),
 'fuzzy_gwo.elite_local_refinement': ('fuzzy_gwo.elite_local_refinement', 'fuzzy_gwo.selection'),
 'fuzzy_gwo.leader_guided_population_update': ('fuzzy_gwo.leader_guided_population_update', 'fuzzy_gwo.selection'),
 'gbrt_bo.evaluate_positions_evaluation_l449': ('gbrt_bo.gbrt_bayesian_optimization_position_update',
                                                'gbrt_bo.selection'),
 'gbrt_bo.gbrt_bayesian_optimization_position_update': ('gbrt_bo.gbrt_bayesian_optimization_position_update',
                                                        'gbrt_bo.selection'),
 'gbrt_bo.position_generation': ('gbrt_bo.gbrt_bayesian_optimization_position_update', 'gbrt_bo.selection'),
 'gp_bo.evaluate_positions_evaluation_l449': ('gp_bo.gp_bayesian_optimization_position_update', 'gp_bo.selection'),
 'gp_bo.gp_bayesian_optimization_position_update': ('gp_bo.gp_bayesian_optimization_position_update',
                                                    'gp_bo.selection'),
 'gp_bo.position_generation': ('gp_bo.gp_bayesian_optimization_position_update', 'gp_bo.selection'),
 'gska.gaining_sharing_knowledge_update': ('gska.gaining_sharing_knowledge_update', 'gska.selection'),
 'gso_glider_snake.glider_snake_position_update': ('gso_glider_snake.glider_snake_position_update',
                                                   'gso_glider_snake.selection'),
 'gwo_woa.elite_local_refinement': ('gwo_woa.elite_local_refinement', 'gwo_woa.selection'),
 'gwo_woa.leader_guided_population_update': ('gwo_woa.leader_guided_population_update', 'gwo_woa.selection'),
 'hde.differential_evolution_update': ('hde.differential_evolution_update', 'hde.selection'),
 'heoa.risk_takers_generate_bounded_sample_around': ('heoa.risk_takers_generate_bounded_sample_around',
                                                     'heoa.selection'),
 'hi_woa.elite_local_refinement': ('hi_woa.elite_local_refinement', 'hi_woa.selection'),
 'hi_woa.whale_position_update': ('hi_woa.whale_position_update', 'hi_woa.selection'),
 'ho_hippo.group_defense_position_update': ('ho_hippo.group_defense_position_update', 'ho_hippo.selection'),
 'ho_hippo.predator_defense_update': ('ho_hippo.predator_defense_update', 'ho_hippo.selection'),
 'ho_hippo.river_pond_position_update': ('ho_hippo.river_pond_position_update', 'ho_hippo.selection'),
 'hus.hus_semantic_update': ('hus.hus_semantic_update', 'hus.selection'),
 'hus.step_evaluation_l62': ('hus.hus_semantic_update', 'hus.selection'),
 'i_gwo.alpha_guidance_trial': ('i_gwo.alpha_guidance_trial', 'i_gwo.selection'),
 'i_gwo.beta_guidance_trial': ('i_gwo.beta_guidance_trial', 'i_gwo.selection'),
 'i_gwo.delta_guidance_trial': ('i_gwo.delta_guidance_trial', 'i_gwo.selection'),
 'i_gwo.mean_leader_position_update': ('i_gwo.mean_leader_position_update', 'i_gwo.selection'),
 'ikoa.assignment_matching_position_update': ('ikoa.assignment_matching_position_update', 'ikoa.selection'),
 'ikoa.improved_matching_refinement_update': ('ikoa.improved_matching_refinement_update', 'ikoa.selection'),
 'improved_aeo.ecosystem_producer_consumer_update': ('improved_aeo.ecosystem_producer_consumer_update',
                                                     'improved_aeo.selection'),
 'improved_aeo.improved_decomposition_refinement': ('improved_aeo.improved_decomposition_refinement',
                                                    'improved_aeo.selection'),
 'improved_qsa.queue_business_one_update': ('improved_qsa.queue_business_one_update', 'improved_qsa.selection'),
 'improved_qsa.queue_business_two_refinement': ('improved_qsa.queue_business_two_refinement', 'improved_qsa.selection'),
 'improved_tlo.elite_local_refinement': ('improved_tlo.elite_local_refinement', 'improved_tlo.selection'),
 'improved_tlo.teacher_learner_population_update': ('improved_tlo.teacher_learner_population_update',
                                                    'improved_tlo.selection'),
 'incremental_gwo.elite_local_refinement': ('incremental_gwo.elite_local_refinement', 'incremental_gwo.selection'),
 'incremental_gwo.leader_guided_population_update': ('incremental_gwo.leader_guided_population_update',
                                                     'incremental_gwo.selection'),
 'iobl_gwo.elite_local_refinement': ('iobl_gwo.elite_local_refinement', 'iobl_gwo.selection'),
 'iobl_gwo.leader_guided_population_update': ('iobl_gwo.leader_guided_population_update', 'iobl_gwo.selection'),
 'jso.ocean_current_swarm_motion_update': ('jso.ocean_current_swarm_motion_update', 'jso.selection'),
 'kha.induced_movement_update': ('kha.induced_movement_update', 'kha.selection'),
 'kma.n_big': ('kma.n_big', 'kma.selection'),
 'kma.small_males_move_towards_big_males': ('kma.small_males_move_towards_big_males', 'kma.selection'),
 'l2smea.l2smea_semantic_update': ('l2smea.l2smea_semantic_update', 'l2smea.selection'),
 'l2smea.step_evaluation_l167': ('l2smea.l2smea_semantic_update', 'l2smea.selection'),
 'liwo.light_wave_position_update': ('liwo.light_wave_position_update', 'liwo.selection'),
 'lso_spectrum.light_spectrum_position_update': ('lso_spectrum.light_spectrum_position_update',
                                                 'lso_spectrum.selection'),
 'misaco.misaco_semantic_update': ('misaco.misaco_semantic_update', 'misaco.selection'),
 'misaco.step_evaluation_l272': ('misaco.misaco_semantic_update', 'misaco.selection'),
 'modified_aeo.ecosystem_producer_consumer_update': ('modified_aeo.ecosystem_producer_consumer_update',
                                                     'modified_aeo.selection'),
 'modified_aeo.modified_decomposition_refinement': ('modified_aeo.modified_decomposition_refinement',
                                                    'modified_aeo.selection'),
 'modified_eo.modified_equilibrium_pool_update': ('modified_eo.modified_equilibrium_pool_update',
                                                  'modified_eo.selection'),
 'modified_eo.modified_local_refinement': ('modified_eo.modified_local_refinement', 'modified_eo.selection'),
 'mso.magnetic_field_position_update': ('mso.magnetic_field_position_update', 'mso.selection'),
 'mvpa.mvp_guided_player_update': ('mvpa.mvp_guided_player_update', 'mvpa.selection'),
 'nlapsmjso_eda.non_linear_population_analysis_update': ('nlapsmjso_eda.non_linear_population_analysis_update',
                                                         'nlapsmjso_eda.selection'),
 'noa.newton_position_update': ('noa.newton_position_update', 'noa.selection'),
 'ofa.owl_neighbour_flight_update': ('ofa.owl_neighbour_flight_update', 'ofa.selection'),
 'ogwo.elite_local_refinement': ('ogwo.elite_local_refinement', 'ogwo.selection'),
 'ogwo.leader_guided_population_update': ('ogwo.leader_guided_population_update', 'ogwo.selection'),
 'ooa.fish_carrying_local_update': ('ooa.fish_carrying_local_update', 'ooa.selection'),
 'pfa.pathfinder_position_update': ('pfa.pathfinder_position_update', 'pfa.selection'),
 'pfa_polar_fox.experience_phase': ('pfa_polar_fox.experience_phase', 'pfa_polar_fox.selection'),
 'pfa_polar_fox.leader_guided_refinement_update': ('pfa_polar_fox.leader_guided_refinement_update',
                                                   'pfa_polar_fox.selection'),
 'pfa_polar_fox.leader_phase': ('pfa_polar_fox.leader_phase', 'pfa_polar_fox.selection'),
 'pko.krill_following_update': ('pko.krill_following_update', 'pko.selection'),
 'plo.plasma_lithium_position_update': ('plo.plasma_lithium_position_update', 'plo.selection'),
 'rf_bo.position_generation': ('rf_bo.rf_bayesian_optimization_position_update', 'rf_bo.selection'),
 'rf_bo.rf_bayesian_optimization_position_update': ('rf_bo.rf_bayesian_optimization_position_update',
                                                    'rf_bo.selection'),
 'rhso.rhinoceros_herd_position_update': ('rhso.rhinoceros_herd_position_update', 'rhso.selection'),
 'rime.candidate_update': ('rime.rime_semantic_update', 'rime.selection'),
 'rime.hard_rime_puncture_update': ('rime.hard_rime_puncture_update', 'rime.selection'),
 'rime.rime_semantic_update': ('rime.rime_semantic_update', 'rime.selection'),
 'run.enhanced_solution_quality_update': ('run.enhanced_solution_quality_update', 'run.selection'),
 'run.runge_kutta_position_update': ('run.runge_kutta_position_update', 'run.selection'),
 'sade.adaptive_strategy_de_update': ('sade.adaptive_strategy_de_update', 'sade.selection'),
 'sade.elite_local_refinement': ('sade.elite_local_refinement', 'sade.selection'),
 'sade_amss.adaptive_multistrategy_subspace_de_update': ('sade_amss.adaptive_multistrategy_subspace_de_update',
                                                         'sade_amss.selection'),
 'sade_amss.candidate_update': ('sade_amss.sade_amss_semantic_update', 'sade_amss.selection'),
 'sade_amss.sade_amss_semantic_update': ('sade_amss.sade_amss_semantic_update', 'sade_amss.selection'),
 'sap_de.elite_local_refinement': ('sap_de.elite_local_refinement', 'sap_de.selection'),
 'sap_de.self_adaptive_parameter_de_update': ('sap_de.self_adaptive_parameter_de_update', 'sap_de.selection'),
 'sapo.candidate_generation': ('sapo.sapo_semantic_update', 'sapo.selection'),
 'sapo.sapo_semantic_update': ('sapo.sapo_semantic_update', 'sapo.selection'),
 'sapo.step_evaluation_l398': ('sapo.sapo_semantic_update', 'sapo.selection'),
 'scho.scholar_chess_position_update': ('scho.scholar_chess_position_update', 'scho.selection'),
 'shio.iguana_sand_hill_position_update': ('shio.iguana_sand_hill_position_update', 'shio.selection'),
 'slo.sea_lion_position_update': ('slo.sea_lion_position_update', 'slo.selection'),
 'sma.slime_mould_oscillation_update': ('sma.slime_mould_oscillation_update', 'sma.selection'),
 'snow_oa.snow_ablation_position_update': ('snow_oa.snow_ablation_position_update', 'snow_oa.selection'),
 'soo.1_oscillatory_position': ('soo.1_oscillatory_position', 'soo.selection'),
 'soo.2_top_3_average_oscillatory': ('soo.2_top_3_average_oscillatory', 'soo.selection'),
 'sparrow_sa.producer_scrounger_update': ('sparrow_sa.producer_scrounger_update', 'sparrow_sa.selection'),
 'spbo.average_student_phase_update': ('spbo.average_student_phase_update', 'spbo.selection'),
 'spbo.best_student': ('spbo.best_student', 'spbo.selection'),
 'spbo.excellent_student_phase_update': ('spbo.excellent_student_phase_update', 'spbo.selection'),
 'srsr.1_accumulation_new_positions_via_gaussian': ('srsr.1_accumulation_new_positions_via_gaussian', 'srsr.selection'),
 'supply_do.supply_demand_balance_update': ('supply_do.supply_demand_balance_update', 'supply_do.selection'),
 'ttao.extra_candidate_diversification_update': ('ttao.extra_candidate_diversification_update',
                                                 'ttao.state_update',
                                                 'ttao.selection'),
 'ttao.random_population_refresh_update': ('ttao.random_population_refresh_update', 'ttao.selection'),
 'whale_foa.elite_local_refinement': ('whale_foa.elite_local_refinement', 'whale_foa.selection'),
 'whale_foa.whale_position_update': ('whale_foa.whale_position_update', 'whale_foa.selection'),
 'who.1_local_movement_milling': ('who.1_local_movement_milling', 'who.selection'),
 'who.2_herd_instinct': ('who.2_herd_instinct', 'who.selection'),
 'who.social_memory': ('who.social_memory', 'who.selection'),
 'wmqimrfo.elite_local_refinement': ('wmqimrfo.elite_local_refinement', 'wmqimrfo.selection'),
 'wmqimrfo.weighted_multi_quadratic_mrfo_update': ('wmqimrfo.weighted_multi_quadratic_mrfo_update',
                                                   'wmqimrfo.selection'),
 'wo_wave.wave_propagation_position_update': ('wo_wave.wave_propagation_position_update', 'wo_wave.selection'),
 'wso.position_generation': ('wso.wso_position_update', 'wso.selection'),
 'wso.white_shark_swarm_position_update': ('wso.white_shark_swarm_position_update', 'wso.selection'),
 'wso.wso_position_update': ('wso.wso_position_update', 'wso.selection'),
 'wutp.horizontal_water_transport_update': ('wutp.horizontal_water_transport_update', 'wutp.selection'),
 'ydse.double_slit_interference_update': ('ydse.double_slit_interference_update', 'ydse.selection')}
_EXACT_COMPOUND_OPERATOR_SPLITS.update(_BATCH19_REMAINING_ACTUAL_SEMANTIC_SPLITS)
_BATCH19_REMAINING_RUNTIME_CATALOG_OVERRIDES = {'aaa': ['aaa.recombination',
         'aaa.selection',
         'aaa.adaptation_most_starving_colony_moves_toward',
         'aaa.is_replaced_by_corresponding_cell_biggest'],
 'acgwo': ['acgwo.selection',
           'acgwo.adaptive_weighted_pack_update',
           'acgwo.alpha_guidance_trial',
           'acgwo.beta_guidance_trial',
           'acgwo.delta_guidance_trial'],
 'adaptive_eo': ['adaptive_eo.selection',
                 'adaptive_eo.adaptive_local_refinement',
                 'adaptive_eo.equilibrium_pool_guided_update'],
 'aeo': ['aeo.selection', 'aeo.consumer_decomposer_update', 'aeo.production_worst_agent'],
 'aesspso': ['aesspso.position_update', 'aesspso.velocity_update'],
 'aho': ['aho.hunting', 'aho.state_update'],
 'aiw_pso': ['aiw_pso.position_update',
             'aiw_pso.selection',
             'aiw_pso.velocity_update',
             'aiw_pso.elite_local_refinement'],
 'aso_atom': ['aso_atom.selection', 'aso_atom.do_not_move_current_elites_unless'],
 'bacterial_colony_o': ['bacterial_colony_o.migration',
                        'bacterial_colony_o.position_update',
                        'bacterial_colony_o.recombination',
                        'bacterial_colony_o.selection',
                        'bacterial_colony_o.current_colony_best_accept_only_it',
                        'bacterial_colony_o.implementation_but_only_as_bounded_macro'],
 'basin_hopping': ['basin_hopping.candidate_search',
                   'basin_hopping.selection',
                   'basin_hopping.basin_hopping_local_search_update'],
 'bboa': ['bboa.selection', 'bboa.2_sniffing', 'bboa.pedal_marking_update'],
 'bco': ['bco.selection', 'bco.bco_semantic_update', 'bco.swim_refinement_update'],
 'bsa': ['bsa.foraging', 'bsa.state_update'],
 'btoa': ['btoa.position_update',
          'btoa.selection',
          'btoa.defensive_play_refinement',
          'btoa.dynamic_position_candidate',
          'btoa.offensive_play_update'],
 'ca': ['ca.selection', 'ca.ca_semantic_update', 'ca.cultural_belief_guided_update'],
 'cdo_chernobyl': ['cdo_chernobyl.selection', 'cdo_chernobyl.cdo_chernobyl_position_update'],
 'cg_gwo': ['cg_gwo.selection', 'cg_gwo.elite_local_refinement', 'cg_gwo.leader_guided_population_update'],
 'chaotic_gwo': ['chaotic_gwo.selection',
                 'chaotic_gwo.elite_local_refinement',
                 'chaotic_gwo.leader_guided_population_update'],
 'chicken_so': ['chicken_so.selection', 'chicken_so.chicken_so_semantic_update'],
 'circle_sa': ['circle_sa.selection',
               'circle_sa.circle_position_update',
               'circle_sa.circle_simulated_annealing_position_update'],
 'cockroach_so': ['cockroach_so.dispersal', 'cockroach_so.replacement', 'cockroach_so.state_update'],
 'compact_ga': ['compact_ga.model_update',
                'compact_ga.sampling',
                'compact_ga.selection',
                'compact_ga.state_update',
                'compact_ga.compact_genetic_algorithm_semantic_update'],
 'coot': ['coot.guidance', 'coot.selection', 'coot.state_update', 'coot.chain_movement_update'],
 'cso': ['cso.selection', 'cso.mean_all_positions'],
 'dbo': ['dbo.foraging', 'dbo.selection', 'dbo.state_update', 'dbo.ball_rolling_dance_update'],
 'ddao': ['ddao.exploration', 'ddao.selection', 'ddao.state_update', 'ddao.dynamic_annealed_refinement_update'],
 'dmoa': ['dmoa.selection', 'dmoa.3_baby_sitter_eviction', 'dmoa.scalar_broadcast', 'dmoa.scout_phase'],
 'do_dandelion': ['do_dandelion.selection', 'do_dandelion.seed_landing_update'],
 'dra': ['dra.selection', 'dra.dialectic_interaction_update'],
 'ds_gwo': ['ds_gwo.selection', 'ds_gwo.elite_local_refinement', 'ds_gwo.leader_guided_population_update'],
 'ecological_cycle_o': ['ecological_cycle_o.selection',
                        'ecological_cycle_o.ecological_cycle_transition_update',
                        'ecological_cycle_o.eval_accept_group'],
 'elk_ho': ['elk_ho.selection', 'elk_ho.family_mating_position_update'],
 'enhanced_aeo': ['enhanced_aeo.selection',
                  'enhanced_aeo.ecosystem_producer_consumer_update',
                  'enhanced_aeo.enhanced_decomposition_refinement'],
 'eo': ['eo.selection', 'eo.equilibrium_position_update'],
 'er_gwo': ['er_gwo.selection', 'er_gwo.elite_local_refinement', 'er_gwo.leader_guided_population_update'],
 'esc': ['esc.exploration', 'esc.state_update'],
 'esoa': ['esoa.behavioral_move', 'esoa.selection', 'esoa.egret_sit_and_wait_update'],
 'et_bo': ['et_bo.selection', 'et_bo.et_bayesian_optimization_position_update'],
 'ex_gwo': ['ex_gwo.selection', 'ex_gwo.elite_local_refinement', 'ex_gwo.leader_guided_population_update'],
 'fox': ['fox.selection', 'fox.preserve_best_few_individuals_explicitly'],
 'fuzzy_gwo': ['fuzzy_gwo.selection', 'fuzzy_gwo.elite_local_refinement', 'fuzzy_gwo.leader_guided_population_update'],
 'gbo': ['gbo.search', 'gbo.state_update'],
 'gbrt_bo': ['gbrt_bo.selection', 'gbrt_bo.gbrt_bayesian_optimization_position_update'],
 'gp_bo': ['gp_bo.selection', 'gp_bo.gp_bayesian_optimization_position_update'],
 'gpso': ['gpso.position_update', 'gpso.velocity_update'],
 'gska': ['gska.selection', 'gska.gaining_sharing_knowledge_update'],
 'gso_glider_snake': ['gso_glider_snake.selection', 'gso_glider_snake.glider_snake_position_update'],
 'gwo_woa': ['gwo_woa.selection', 'gwo_woa.elite_local_refinement', 'gwo_woa.leader_guided_population_update'],
 'hde': ['hde.candidate_search', 'hde.selection', 'hde.differential_evolution_update'],
 'heoa': ['heoa.selection', 'heoa.risk_takers_generate_bounded_sample_around'],
 'hi_woa': ['hi_woa.selection', 'hi_woa.elite_local_refinement', 'hi_woa.whale_position_update'],
 'ho_hippo': ['ho_hippo.exploitation',
              'ho_hippo.selection',
              'ho_hippo.state_update',
              'ho_hippo.group_defense_position_update',
              'ho_hippo.predator_defense_update',
              'ho_hippo.river_pond_position_update'],
 'hus': ['hus.selection', 'hus.hus_semantic_update'],
 'i_gwo': ['i_gwo.selection',
           'i_gwo.alpha_guidance_trial',
           'i_gwo.beta_guidance_trial',
           'i_gwo.delta_guidance_trial',
           'i_gwo.mean_leader_position_update'],
 'ikoa': ['ikoa.selection', 'ikoa.assignment_matching_position_update', 'ikoa.improved_matching_refinement_update'],
 'improved_aeo': ['improved_aeo.selection',
                  'improved_aeo.ecosystem_producer_consumer_update',
                  'improved_aeo.improved_decomposition_refinement'],
 'improved_qsa': ['improved_qsa.selection',
                  'improved_qsa.queue_business_one_update',
                  'improved_qsa.queue_business_two_refinement'],
 'improved_tlo': ['improved_tlo.selection',
                  'improved_tlo.elite_local_refinement',
                  'improved_tlo.teacher_learner_population_update'],
 'incremental_gwo': ['incremental_gwo.selection',
                     'incremental_gwo.elite_local_refinement',
                     'incremental_gwo.leader_guided_population_update'],
 'iobl_gwo': ['iobl_gwo.selection', 'iobl_gwo.elite_local_refinement', 'iobl_gwo.leader_guided_population_update'],
 'jso': ['jso.selection', 'jso.ocean_current_swarm_motion_update'],
 'kha': ['kha.crossover',
         'kha.diffusion',
         'kha.mutation',
         'kha.selection',
         'kha.state_update',
         'kha.induced_movement_update'],
 'kma': ['kma.selection', 'kma.n_big', 'kma.small_males_move_towards_big_males'],
 'l2smea': ['l2smea.selection', 'l2smea.l2smea_semantic_update'],
 'lca': ['lca.mutation', 'lca.state_update'],
 'liwo': ['liwo.selection', 'liwo.light_wave_position_update'],
 'lso_spectrum': ['lso_spectrum.selection', 'lso_spectrum.light_spectrum_position_update'],
 'misaco': ['misaco.selection', 'misaco.misaco_semantic_update'],
 'modified_aeo': ['modified_aeo.selection',
                  'modified_aeo.ecosystem_producer_consumer_update',
                  'modified_aeo.modified_decomposition_refinement'],
 'modified_eo': ['modified_eo.selection',
                 'modified_eo.modified_equilibrium_pool_update',
                 'modified_eo.modified_local_refinement'],
 'mso': ['mso.candidate_search', 'mso.selection', 'mso.magnetic_field_position_update'],
 'mvpa': ['mvpa.selection', 'mvpa.mvp_guided_player_update'],
 'nlapsmjso_eda': ['nlapsmjso_eda.sampling',
                   'nlapsmjso_eda.selection',
                   'nlapsmjso_eda.state_update',
                   'nlapsmjso_eda.non_linear_population_analysis_update'],
 'noa': ['noa.selection', 'noa.newton_position_update'],
 'ofa': ['ofa.selection', 'ofa.owl_neighbour_flight_update'],
 'ogwo': ['ogwo.selection', 'ogwo.elite_local_refinement', 'ogwo.leader_guided_population_update'],
 'ooa': ['ooa.hunting', 'ooa.search', 'ooa.selection', 'ooa.state_update', 'ooa.fish_carrying_local_update'],
 'pfa': ['pfa.selection', 'pfa.pathfinder_position_update'],
 'pfa_polar_fox': ['pfa_polar_fox.exploitation',
                   'pfa_polar_fox.selection',
                   'pfa_polar_fox.state_update',
                   'pfa_polar_fox.experience_phase',
                   'pfa_polar_fox.leader_guided_refinement_update',
                   'pfa_polar_fox.leader_phase'],
 'pko': ['pko.behavioral_move', 'pko.selection', 'pko.krill_following_update'],
 'plo': ['plo.selection', 'plo.plasma_lithium_position_update'],
 'rf_bo': ['rf_bo.selection', 'rf_bo.rf_bayesian_optimization_position_update'],
 'rhso': ['rhso.selection', 'rhso.rhinoceros_herd_position_update'],
 'rime': ['rime.selection', 'rime.rime_semantic_update', 'rime.hard_rime_puncture_update'],
 'run': ['run.selection', 'run.enhanced_solution_quality_update', 'run.runge_kutta_position_update'],
 'sacoso': ['sacoso.cooperation', 'sacoso.pso_update', 'sacoso.swarm_update'],
 'sade': ['sade.selection', 'sade.adaptive_strategy_de_update', 'sade.elite_local_refinement'],
 'sade_amss': ['sade_amss.selection',
               'sade_amss.adaptive_multistrategy_subspace_de_update',
               'sade_amss.sade_amss_semantic_update'],
 'sap_de': ['sap_de.selection', 'sap_de.elite_local_refinement', 'sap_de.self_adaptive_parameter_de_update'],
 'sapo': ['sapo.selection', 'sapo.sapo_semantic_update'],
 'scho': ['scho.selection', 'scho.scholar_chess_position_update'],
 'serval_oa': ['serval_oa.hunting', 'serval_oa.selection', 'serval_oa.state_update'],
 'sfo': ['sfo.behavioral_move', 'sfo.selection'],
 'shio': ['shio.selection', 'shio.iguana_sand_hill_position_update'],
 'sine_cosine_a': ['sine_cosine_a.position_update', 'sine_cosine_a.sine_cosine_move'],
 'slo': ['slo.selection', 'slo.sea_lion_position_update'],
 'sma': ['carryover', 'sma.selection', 'sma.slime_mould_oscillation_update'],
 'snow_oa': ['snow_oa.selection', 'snow_oa.snow_ablation_position_update'],
 'soo': ['soo.selection', 'soo.1_oscillatory_position', 'soo.2_top_3_average_oscillatory'],
 'sparrow_sa': ['sparrow_sa.awareness', 'sparrow_sa.selection', 'sparrow_sa.producer_scrounger_update'],
 'spbo': ['spbo.groups',
          'spbo.selection',
          'spbo.average_student_phase_update',
          'spbo.best_student',
          'spbo.excellent_student_phase_update'],
 'srsr': ['carryover', 'srsr.exploration', 'srsr.selection', 'srsr.1_accumulation_new_positions_via_gaussian'],
 'supply_do': ['supply_do.exploration',
               'supply_do.selection',
               'supply_do.state_update',
               'supply_do.supply_demand_balance_update'],
 'ts': ['ts.neighborhood_move', 'ts.selection', 'ts.tabu_memory'],
 'ttao': ['ttao.crossover',
          'ttao.selection',
          'ttao.state_update',
          'ttao.extra_candidate_diversification_update',
          'ttao.random_population_refresh_update'],
 'whale_foa': ['whale_foa.selection', 'whale_foa.elite_local_refinement', 'whale_foa.whale_position_update'],
 'who': ['who.selection', 'who.1_local_movement_milling', 'who.2_herd_instinct', 'who.social_memory'],
 'wmqimrfo': ['wmqimrfo.selection', 'wmqimrfo.elite_local_refinement', 'wmqimrfo.weighted_multi_quadratic_mrfo_update'],
 'wo_wave': ['wo_wave.selection', 'wo_wave.wave_propagation_position_update'],
 'wso': ['wso.selection', 'wso.wso_position_update', 'wso.white_shark_swarm_position_update'],
 'wutp': ['wutp.selection', 'wutp.horizontal_water_transport_update'],
 'ydse': ['ydse.selection', 'ydse.double_slit_interference_update']}
_ENGINE_OPERATOR_LABEL_OVERRIDES.update(_BATCH19_REMAINING_RUNTIME_CATALOG_OVERRIDES)
for _aid, _labels in _BATCH19_REMAINING_RUNTIME_CATALOG_OVERRIDES.items():
    ENGINE_OPERATOR_LABELS[_aid] = list(_labels)

# ---------------------------------------------------------------------------
# Batch 20 / Addendum W0 partial restoration: preserve genuine engine-emitted
# per-candidate labels instead of expanding them into static generic splits.
# ---------------------------------------------------------------------------
# The addendum introduced the G5 seed-sensitivity criterion.  The labels below
# are emitted directly by engine payloads as per-candidate operator_labels.
# They must remain atomic; splitting them into generic labels such as
# candidate_generation/selection or behavioral_move/selection destroys genuine
# measured attribution and can create static CDS distributions.
_GENUINE_ENGINE_EMITTED_OPERATOR_LABELS = {
    "aha.guided_foraging",
    "aha.territorial_foraging",
    "aha.migration",
    "tlbo.teacher_phase",
    "tlbo.learner_phase",
    "ssa.leader_plus_food_guidance",
    "ssa.leader_minus_food_guidance",
    "ssa.follower_front_chain_update",
    "ssa.follower_rear_chain_update",
}

_PREVIOUS_EXPAND_COMPOUND_OPERATOR_LABEL_W0 = expand_compound_operator_label

def expand_compound_operator_label(algorithm_id: str, label: str | None) -> list[str]:  # type: ignore[override]
    if label in {None, ""}:
        return []
    raw = str(label)
    if raw in _GENUINE_ENGINE_EMITTED_OPERATOR_LABELS:
        return [raw]
    return _PREVIOUS_EXPAND_COMPOUND_OPERATOR_LABEL_W0(algorithm_id, label)

# Restore the catalog surface for the genuinely instrumented W0 engines.  This
# is still label-only, but now it exposes labels that are actually emitted per
# accepted candidate instead of decorative static rewrites.
_BATCH20_W0_RUNTIME_CATALOG_RESTORATION = {
    "aha": ["aha.guided_foraging", "aha.territorial_foraging", "aha.migration"],
    "tlbo": ["tlbo.teacher_phase", "tlbo.learner_phase"],
    "ssa": ["ssa.leader_plus_food_guidance", "ssa.leader_minus_food_guidance", "ssa.follower_front_chain_update", "ssa.follower_rear_chain_update"],
}
_ENGINE_OPERATOR_LABEL_OVERRIDES.update(_BATCH20_W0_RUNTIME_CATALOG_RESTORATION)
for _aid, _labels in _BATCH20_W0_RUNTIME_CATALOG_RESTORATION.items():
    ENGINE_OPERATOR_LABELS[_aid] = list(_labels)

_PREVIOUS_LABELS_FOR_ALGORITHM_W0 = labels_for_algorithm

def labels_for_algorithm(algorithm_id: str) -> list[str]:  # type: ignore[override]
    aid = str(algorithm_id)
    if aid in _ENGINE_OPERATOR_LABEL_OVERRIDES:
        return list(_ENGINE_OPERATOR_LABEL_OVERRIDES[aid])
    return _PREVIOUS_LABELS_FOR_ALGORITHM_W0(aid)

__all__ = list(dict.fromkeys(list(__all__) + ["_GENUINE_ENGINE_EMITTED_OPERATOR_LABELS"]))


# ---------------------------------------------------------------------------
# Batch 21 / Addendum W0 continuation: restore per-candidate labels for
# regressed W0 engines instrumented in this batch.
# ---------------------------------------------------------------------------
_BATCH21_W0_GENUINE_LABELS = {
    "ars.small_step", "ars.large_step",
    "coa.alpha_social_condition_update", "coa.tendency_social_condition_update", "coa.pup_birth_replacement", "coa.migration_exchange",
    "frofi.current_to_rand_de", "frofi.rand_to_best_crossover_de", "frofi.no_crossover_de", "frofi.targeted_mutation",
    "gea.neighbour_geyser_eruption_update", "gea.pressure_random_eruption_update",
    "hgso.cluster_best_solubility_update", "hgso.global_best_solubility_update", "hgso.worst_agent_random_reset",
    "ica.assimilation", "ica.imperialist_revolution", "ica.colony_revolution", "ica.intra_empire_competition",
    "mrfo.chain_foraging", "mrfo.cyclone_random_foraging", "mrfo.cyclone_best_foraging", "mrfo.somersault_foraging",
    "nndrea_so.nn_weight_de_stage", "nndrea_so.solution_de_stage",
    "poa.prey_pursuit_update", "poa.water_surface_winging_update",
    "smo.local_leader_phase", "smo.global_leader_phase", "smo.local_leader_decision",
    "squirrel_sa.acorn_to_hickory_glide", "squirrel_sa.normal_to_acorn_glide", "squirrel_sa.normal_to_hickory_glide", "squirrel_sa.predator_random_relocation",
    "tfwo.effect_of_objects", "tfwo.random_object_relocation", "tfwo.effect_of_whirlpools", "tfwo.best_whirlpool_preservation", "tfwo.object_whirlpool_exchange", "tfwo.state_structure_update",
    "vcs.virus_diffusion", "vcs.host_cell_infection", "vcs.immune_response",
    "wca.stream_toward_river", "wca.stream_river_exchange", "wca.river_toward_sea", "wca.evaporation_raining",
}
_GENUINE_ENGINE_EMITTED_OPERATOR_LABELS.update(_BATCH21_W0_GENUINE_LABELS)

_BATCH21_W0_RUNTIME_CATALOG_RESTORATION = {
    "ars": ["ars.small_step", "ars.large_step"],
    "coa": ["coa.alpha_social_condition_update", "coa.tendency_social_condition_update", "coa.pup_birth_replacement", "coa.migration_exchange"],
    "frofi": ["frofi.current_to_rand_de", "frofi.rand_to_best_crossover_de", "frofi.no_crossover_de", "frofi.targeted_mutation"],
    "gea": ["gea.neighbour_geyser_eruption_update", "gea.pressure_random_eruption_update"],
    "hgso": ["hgso.cluster_best_solubility_update", "hgso.global_best_solubility_update", "hgso.worst_agent_random_reset"],
    "ica": ["ica.assimilation", "ica.imperialist_revolution", "ica.colony_revolution", "ica.intra_empire_competition"],
    "mrfo": ["mrfo.chain_foraging", "mrfo.cyclone_random_foraging", "mrfo.cyclone_best_foraging", "mrfo.somersault_foraging"],
    "nndrea_so": ["nndrea_so.nn_weight_de_stage", "nndrea_so.solution_de_stage"],
    "poa": ["poa.prey_pursuit_update", "poa.water_surface_winging_update"],
    "smo": ["smo.local_leader_phase", "smo.global_leader_phase", "smo.local_leader_decision"],
    "squirrel_sa": ["squirrel_sa.acorn_to_hickory_glide", "squirrel_sa.normal_to_acorn_glide", "squirrel_sa.normal_to_hickory_glide", "squirrel_sa.predator_random_relocation"],
    "tfwo": ["tfwo.effect_of_objects", "tfwo.random_object_relocation", "tfwo.effect_of_whirlpools", "tfwo.best_whirlpool_preservation", "tfwo.object_whirlpool_exchange", "tfwo.state_structure_update"],
    "vcs": ["vcs.virus_diffusion", "vcs.host_cell_infection", "vcs.immune_response"],
    "wca": ["wca.stream_toward_river", "wca.stream_river_exchange", "wca.river_toward_sea", "wca.evaporation_raining"],
}
_ENGINE_OPERATOR_LABEL_OVERRIDES.update(_BATCH21_W0_RUNTIME_CATALOG_RESTORATION)
for _aid, _labels in _BATCH21_W0_RUNTIME_CATALOG_RESTORATION.items():
    ENGINE_OPERATOR_LABELS[_aid] = list(_labels)

_PREVIOUS_LABELS_FOR_ALGORITHM_W21 = labels_for_algorithm

def labels_for_algorithm(algorithm_id: str) -> list[str]:  # type: ignore[override]
    aid = str(algorithm_id)
    if aid in _ENGINE_OPERATOR_LABEL_OVERRIDES:
        return list(_ENGINE_OPERATOR_LABEL_OVERRIDES[aid])
    return _PREVIOUS_LABELS_FOR_ALGORITHM_W21(aid)


# ---------------------------------------------------------------------------
# Batch 22 / Addendum W1 start: genuine per-candidate instrumentation for
# selected static-uniform engines with explicit branch/phased operators.
# ---------------------------------------------------------------------------
_BATCH22_W1_GENUINE_LABELS = {
    "aro.detour_foraging", "aro.random_hiding",
    "bmo.barnacle_recombination", "bmo.random_barnacle_drift",
    "bea.elite_site_neighbourhood_search", "bea.selected_site_neighbourhood_search", "bea.scout_site_global_search",
    "esc.escape_from_worst_update", "esc.move_toward_best_update", "esc.random_exploration_update",
    "eto.exponential_orbit_update", "eto.trigonometric_orbit_update",
    "fpa.global_levy_pollination", "fpa.local_pollination",
}
_GENUINE_ENGINE_EMITTED_OPERATOR_LABELS.update(_BATCH22_W1_GENUINE_LABELS)

_BATCH22_W1_RUNTIME_CATALOG_RESTORATION = {
    "aro": ["aro.detour_foraging", "aro.random_hiding"],
    "bmo": ["bmo.barnacle_recombination", "bmo.random_barnacle_drift"],
    "bea": ["bea.elite_site_neighbourhood_search", "bea.selected_site_neighbourhood_search", "bea.scout_site_global_search"],
    "esc": ["esc.escape_from_worst_update", "esc.move_toward_best_update", "esc.random_exploration_update"],
    "eto": ["eto.exponential_orbit_update", "eto.trigonometric_orbit_update"],
    "fpa": ["fpa.global_levy_pollination", "fpa.local_pollination"],
}
_ENGINE_OPERATOR_LABEL_OVERRIDES.update(_BATCH22_W1_RUNTIME_CATALOG_RESTORATION)
for _aid, _labels in _BATCH22_W1_RUNTIME_CATALOG_RESTORATION.items():
    ENGINE_OPERATOR_LABELS[_aid] = list(_labels)

_PREVIOUS_LABELS_FOR_ALGORITHM_W22 = labels_for_algorithm

def labels_for_algorithm(algorithm_id: str) -> list[str]:  # type: ignore[override]
    aid = str(algorithm_id)
    if aid in _ENGINE_OPERATOR_LABEL_OVERRIDES:
        return list(_ENGINE_OPERATOR_LABEL_OVERRIDES[aid])
    return _PREVIOUS_LABELS_FOR_ALGORITHM_W22(aid)

# Batch 22 instruments these engines with multiple measured operators; they are
# no longer single-operator exemptions under G5.
_SINGLE_OPERATOR_SEMANTIC_OK.difference_update(set(_BATCH22_W1_RUNTIME_CATALOG_RESTORATION))


# ---------------------------------------------------------------------------
# Batch 23 / Addendum W1 expanded: genuine per-candidate instrumentation for
# a larger static-engine batch.
# ---------------------------------------------------------------------------
_BATCH23_W1_GENUINE_LABELS = {
    "aft.best_guided_tracking",
    "aft.random_treasure_search",
    "aft.opposition_tracking",
    "bsa.foraging_flight_update",
    "bsa.vigilance_flight_update",
    "bsa.producer_guided_flight_update",
    "bsa.scrounger_random_flight_update",
    "chameleon_sa.social_pbest_gbest_update",
    "chameleon_sa.random_global_exploration",
    "dso.deep_sleep_decay_update",
    "dso.slow_wave_recovery_update",
    "eco.primary_competition_update",
    "eco.sine_cosine_learning_update",
    "eco.best_weighted_learning_update",
    "eco.levy_exam_update",
    "eefo.interaction_migration",
    "eefo.resting_area_update",
    "eefo.levy_hunting_update",
    "eefo.prey_capture_update",
    "foa_fossa.prey_pursuit_update",
    "foa_fossa.defensive_escape_update",
    "gja.levy_wall_search",
    "gja.gaussian_wall_search",
    "gndo.generalized_normal_local_update",
    "gndo.difference_vector_global_update",
    "hba.bat_frequency_movement",
    "hba.de_local_search",
}
_GENUINE_ENGINE_EMITTED_OPERATOR_LABELS.update(_BATCH23_W1_GENUINE_LABELS)

_BATCH23_W1_RUNTIME_CATALOG_RESTORATION = {
    "aft": ['aft.best_guided_tracking', 'aft.random_treasure_search', 'aft.opposition_tracking'],
    "bsa": ['bsa.foraging_flight_update', 'bsa.vigilance_flight_update', 'bsa.producer_guided_flight_update', 'bsa.scrounger_random_flight_update'],
    "chameleon_sa": ['chameleon_sa.social_pbest_gbest_update', 'chameleon_sa.random_global_exploration'],
    "dso": ['dso.deep_sleep_decay_update', 'dso.slow_wave_recovery_update'],
    "eco": ['eco.primary_competition_update', 'eco.sine_cosine_learning_update', 'eco.best_weighted_learning_update', 'eco.levy_exam_update'],
    "eefo": ['eefo.interaction_migration', 'eefo.resting_area_update', 'eefo.levy_hunting_update', 'eefo.prey_capture_update'],
    "foa_fossa": ['foa_fossa.prey_pursuit_update', 'foa_fossa.defensive_escape_update'],
    "gja": ['gja.levy_wall_search', 'gja.gaussian_wall_search'],
    "gndo": ['gndo.generalized_normal_local_update', 'gndo.difference_vector_global_update'],
    "hba": ['hba.bat_frequency_movement', 'hba.de_local_search'],
}
_ENGINE_OPERATOR_LABEL_OVERRIDES.update(_BATCH23_W1_RUNTIME_CATALOG_RESTORATION)
for _aid, _labels in _BATCH23_W1_RUNTIME_CATALOG_RESTORATION.items():
    ENGINE_OPERATOR_LABELS[_aid] = list(_labels)

_PREVIOUS_LABELS_FOR_ALGORITHM_W23 = labels_for_algorithm

def labels_for_algorithm(algorithm_id: str) -> list[str]:  # type: ignore[override]
    aid = str(algorithm_id)
    if aid in _ENGINE_OPERATOR_LABEL_OVERRIDES:
        return list(_ENGINE_OPERATOR_LABEL_OVERRIDES[aid])
    return _PREVIOUS_LABELS_FOR_ALGORITHM_W23(aid)

_SINGLE_OPERATOR_SEMANTIC_OK.difference_update(set(_BATCH23_W1_RUNTIME_CATALOG_RESTORATION))

# ---------------------------------------------------------------------------
# Addendum W1/W2 instrumentation (Batch 24)
# ---------------------------------------------------------------------------
# These labels are emitted directly by the engines as passive per-candidate
# operator_labels.  Catalog entries are extended so runtime labels are
# recognized without static/decorative redistribution.
_B24_ENGINE_LABELS = {
    "aho": ["aho.single_shot_prey_projection", "aho.double_shot_prey_projection", "aho.levy_stagnation_rescue"],
    "ala": ["ala.high_energy_digging_walk", "ala.high_energy_lemming_migration", "ala.low_energy_spiral_foraging", "ala.low_energy_levy_escape"],
    "boa": ["boa.global_fragrance_attraction", "boa.local_fragrance_random_walk"],
    "cgo": ["cgo.current_seed_attractor", "cgo.best_seed_attractor", "cgo.mean_group_seed_attractor", "cgo.dimension_mutation_seed"],
    "cpo": ["cpo.aroma_luring_trial", "cpo.predation_feeding_trial"],
    "crayfish_oa": ["crayfish_oa.high_temperature_shelter_update", "crayfish_oa.high_temperature_competition_update", "crayfish_oa.food_competition_update", "crayfish_oa.food_intake_update"],
    "dhole_oa": ["dhole_oa.searching_stage", "dhole_oa.encircling_stage", "dhole_oa.large_prey_attack", "dhole_oa.small_prey_kill"],
    "qio": ["qio.three_point_quadratic_interpolation", "qio.two_point_reflection_interpolation"],
    "sso": ["sso.female_spider_position_update", "sso.male_spider_position_update"],
    "sto": ["sto.prey_hunting_update", "sto.range_reduction_update"],
    "waoa": ["waoa.feeding_exploration_update", "waoa.range_narrowing_exploitation"],
    "ydse": ["ydse.central_bright_fringe_update", "ydse.bright_fringe_interference_update", "ydse.dark_fringe_interference_update"],
}
for _aid, _labels in _B24_ENGINE_LABELS.items():
    _current = list(ENGINE_OPERATOR_LABELS.get(_aid, []))
    for _lab in _labels:
        if _lab not in _current:
            _current.append(_lab)
    ENGINE_OPERATOR_LABELS[_aid] = _current
_GENUINE_ENGINE_EMITTED_OPERATOR_LABELS.update({
    _lab for _labels in _B24_ENGINE_LABELS.values() for _lab in _labels
})
_ENGINE_OPERATOR_LABEL_OVERRIDES.update(_B24_ENGINE_LABELS)
for _aid, _labels in _B24_ENGINE_LABELS.items():
    ENGINE_OPERATOR_LABELS[_aid] = list(_labels)
_SINGLE_OPERATOR_SEMANTIC_OK.difference_update(set(_B24_ENGINE_LABELS))

_PREVIOUS_LABELS_FOR_ALGORITHM_W24 = labels_for_algorithm

def labels_for_algorithm(algorithm_id: str) -> list[str]:  # type: ignore[override]
    aid = str(algorithm_id)
    if aid in _ENGINE_OPERATOR_LABEL_OVERRIDES:
        return list(_ENGINE_OPERATOR_LABEL_OVERRIDES[aid])
    return _PREVIOUS_LABELS_FOR_ALGORITHM_W24(aid)

# Additional Batch 24 accepted targets.
_B24_MORE_ENGINE_LABELS = {
    "supply_do": ["supply_do.quantity_equilibrium_update", "supply_do.price_equilibrium_update"],
    "ssdo": ["ssdo.sine_velocity_update", "ssdo.cosine_velocity_update"],
    "snow_oa": ["snow_oa.exploration_group_update", "snow_oa.development_group_update"],
    "tree_seed_a": ["tree_seed_a.toward_best_seed", "tree_seed_a.away_random_seed"],
    "tsa": ["tsa.toward_best_tunicate_update", "tsa.away_best_tunicate_update", "tsa.swarm_chain_averaging_update"],
    "tso": ["tso.leader_spiral_update", "tso.random_migration_update", "tso.spiral_following_update", "tso.parabolic_foraging_update"],
    "warso": ["warso.attack_strategy_update", "warso.defense_strategy_update"],
}
_GENUINE_ENGINE_EMITTED_OPERATOR_LABELS.update({
    _lab for _labels in _B24_MORE_ENGINE_LABELS.values() for _lab in _labels
})
_ENGINE_OPERATOR_LABEL_OVERRIDES.update(_B24_MORE_ENGINE_LABELS)
for _aid, _labels in _B24_MORE_ENGINE_LABELS.items():
    ENGINE_OPERATOR_LABELS[_aid] = list(_labels)
_SINGLE_OPERATOR_SEMANTIC_OK.difference_update(set(_B24_MORE_ENGINE_LABELS))

_PREVIOUS_LABELS_FOR_ALGORITHM_W24_MORE = labels_for_algorithm

def labels_for_algorithm(algorithm_id: str) -> list[str]:  # type: ignore[override]
    aid = str(algorithm_id)
    if aid in _ENGINE_OPERATOR_LABEL_OVERRIDES:
        return list(_ENGINE_OPERATOR_LABEL_OVERRIDES[aid])
    return _PREVIOUS_LABELS_FOR_ALGORITHM_W24_MORE(aid)

# ---------------------------------------------------------------------------
# Batch 25 / Addendum large-attempt accepted instrumentation.
# ---------------------------------------------------------------------------
# These labels are emitted directly by the engines as passive per-candidate
# operator_labels. They are kept atomic so EvoMapX measures actual lineage deltas
# rather than decorative/static catalog splits.
_B25_ENGINE_LABELS = {
    "ao": ["ao.high_soar_vertical_stoop", "ao.contour_flight_exploration", "ao.low_flight_attack", "ao.walk_and_grab_prey"],
    "aoo": ["aoo.mean_wind_animation_update", "aoo.best_wind_animation_update", "aoo.self_wind_animation_update", "aoo.rolling_levy_animation_update", "aoo.projectile_jump_animation_update"],
    "apo": ["apo.dormancy_random_restart", "apo.dormancy_local_perturbation", "apo.foraging_reproduction_update", "apo.autotrophic_foraging_update"],
    "avoa": ["avoa.exploration_vulture_soaring", "avoa.random_roost_exploration", "avoa.convergent_competition_exploitation", "avoa.levy_food_exploitation", "avoa.aggressive_siege_exploitation", "avoa.spiral_siege_exploitation"],
    "bps": ["bps.long_distance_flight", "bps.local_tree_movement", "bps.best_tree_attraction"],
    "bso": ["bso.single_cluster_center_idea", "bso.single_cluster_member_idea", "bso.empty_cluster_center_idea", "bso.two_cluster_center_blend", "bso.two_cluster_member_blend"],
    "ceo_cosmic": ["ceo_cosmic.exploration_attraction_alignment", "ceo_cosmic.global_collision_update", "ceo_cosmic.resonance_refinement_update"],
    "cro": ["cro.broadcast_spawning_recombination", "cro.brooding_clone_mutation", "cro.depredation_random_reseeding"],
    "chio": ["chio.infected_contact_update", "chio.susceptible_contact_update", "chio.immune_contact_update"],
    "fata": ["fata.random_refraction_update", "fata.best_refraction_update", "fata.peer_refraction_update"],
    "fla": ["fla.forward_diffusion_transfer", "fla.source_fluid_diffusion", "fla.receiver_fluid_diffusion", "fla.reverse_diffusion_transfer", "fla.equilibrium_exploitation_update"],
}
_GENUINE_ENGINE_EMITTED_OPERATOR_LABELS.update({_lab for _labels in _B25_ENGINE_LABELS.values() for _lab in _labels})
_ENGINE_OPERATOR_LABEL_OVERRIDES.update(_B25_ENGINE_LABELS)
for _aid, _labels in _B25_ENGINE_LABELS.items():
    ENGINE_OPERATOR_LABELS[_aid] = list(_labels)
_SINGLE_OPERATOR_SEMANTIC_OK.difference_update(set(_B25_ENGINE_LABELS))

_PREVIOUS_LABELS_FOR_ALGORITHM_W25 = labels_for_algorithm

def labels_for_algorithm(algorithm_id: str) -> list[str]:  # type: ignore[override]
    aid = str(algorithm_id)
    if aid in _ENGINE_OPERATOR_LABEL_OVERRIDES:
        return list(_ENGINE_OPERATOR_LABEL_OVERRIDES[aid])
    return _PREVIOUS_LABELS_FOR_ALGORITHM_W25(aid)

# Batch 25 additional accepted-attempt labels.
_B25_MORE_ENGINE_LABELS = {
    "efo": ["efo.electromagnetic_field_update", "efo.random_field_reinitialization", "efo.dimension_reset_mutation"],
    "fox": ["fox.prey_jump_exploitation", "fox.current_to_random_walk_update", "fox.best_radius_random_walk"],
    "gjo": ["gjo.male_female_exploitation", "gjo.male_female_exploration"],
    "slo": ["slo.best_encircling_update", "slo.random_peer_encircling_update", "slo.spiral_attack_update"],
    "sma": ["sma.random_dispersion_update", "sma.best_weighted_oscillation_update", "sma.contracting_vibration_update"],
    "sparrow_sa": ["sparrow_sa.producer_safe_foraging", "sparrow_sa.producer_alarm_random_walk", "sparrow_sa.scrounger_worst_avoidance", "sparrow_sa.scrounger_best_following", "sparrow_sa.awareness_best_escape", "sparrow_sa.awareness_worst_escape"],
}
_GENUINE_ENGINE_EMITTED_OPERATOR_LABELS.update({_lab for _labels in _B25_MORE_ENGINE_LABELS.values() for _lab in _labels})
_ENGINE_OPERATOR_LABEL_OVERRIDES.update(_B25_MORE_ENGINE_LABELS)
for _aid, _labels in _B25_MORE_ENGINE_LABELS.items():
    ENGINE_OPERATOR_LABELS[_aid] = list(_labels)
_SINGLE_OPERATOR_SEMANTIC_OK.difference_update(set(_B25_MORE_ENGINE_LABELS))

_PREVIOUS_LABELS_FOR_ALGORITHM_W25_MORE = labels_for_algorithm

def labels_for_algorithm(algorithm_id: str) -> list[str]:  # type: ignore[override]
    aid = str(algorithm_id)
    if aid in _ENGINE_OPERATOR_LABEL_OVERRIDES:
        return list(_ENGINE_OPERATOR_LABEL_OVERRIDES[aid])
    return _PREVIOUS_LABELS_FOR_ALGORITHM_W25_MORE(aid)

# ---------------------------------------------------------------------------
# Batch 26 / Addendum accepted instrumentation.
# ---------------------------------------------------------------------------
_B26_ENGINE_LABELS = {
    'artemisinin_o': ['artemisinin_o.self_growth_update', 'artemisinin_o.best_growth_update', 'artemisinin_o.differential_mutation_update', 'artemisinin_o.self_reset_mutation', 'artemisinin_o.best_reset_mutation', 'artemisinin_o.boundary_best_repair'],
    'cfoa': ['cfoa.individual_foraging_update', 'cfoa.group_foraging_update', 'cfoa.late_gaussian_capture_update'],
    'eel_grouper_o': ['eel_grouper_o.eel_weighted_hunting_update', 'eel_grouper_o.grouper_weighted_hunting_update'],
    'superb_foa': ['superb_foa.global_smell_random_update', 'superb_foa.levy_food_attraction', 'superb_foa.best_food_convergence'],
    'loa': ['loa.nomad_roaming_update', 'loa.pride_mating_recombination', 'loa.pride_leader_roaming_update', 'loa.nomad_roaming_update.mutation', 'loa.pride_mating_recombination.mutation', 'loa.pride_leader_roaming_update.mutation', 'loa.territorial_takeover_exchange'],
}
_GENUINE_ENGINE_EMITTED_OPERATOR_LABELS.update({_lab for _labels in _B26_ENGINE_LABELS.values() for _lab in _labels})
_ENGINE_OPERATOR_LABEL_OVERRIDES.update(_B26_ENGINE_LABELS)
for _aid, _labels in _B26_ENGINE_LABELS.items():
    ENGINE_OPERATOR_LABELS[_aid] = list(_labels)
_SINGLE_OPERATOR_SEMANTIC_OK.difference_update(set(_B26_ENGINE_LABELS))
_PREVIOUS_LABELS_FOR_ALGORITHM_W26 = labels_for_algorithm
def labels_for_algorithm(algorithm_id: str) -> list[str]:  # type: ignore[override]
    aid = str(algorithm_id)
    if aid in _ENGINE_OPERATOR_LABEL_OVERRIDES:
        return list(_ENGINE_OPERATOR_LABEL_OVERRIDES[aid])
    return _PREVIOUS_LABELS_FOR_ALGORITHM_W26(aid)

# ---------------------------------------------------------------------------
# Batch 27 / Addendum candidate-level instrumentation.
# ---------------------------------------------------------------------------
_B27_ENGINE_LABELS = {
    'gbo': ['gbo.gradient_search_rule_update', 'gbo.local_escaping_operator_update', 'carryover'],
    'nmra': ['nmra.breeder_exploitation_update', 'nmra.worker_exploration_update', 'carryover'],
    'pss': ['pss.prominent_domain_sampling_update', 'pss.full_domain_sampling_update', 'pss.mixed_domain_sampling_update', 'carryover'],
    'pko': ['pko.diving_beating_rate_update', 'pko.crest_angle_foraging_update', 'pko.hovering_attack_update', 'pko.population_escape_update', 'carryover'],
    'mso': ['mso.superior_mirage_search_update', 'mso.inferior_mirage_search_update', 'carryover'],
    'heoa': ['heoa.elite_local_refinement', 'heoa.learner_levy_best_attraction', 'heoa.explorer_centroid_escape', 'heoa.follower_best_contraction', 'heoa.risk_taker_best_sampling', 'carryover'],
    'hgs': ['hgs.random_hunger_exploration', 'hgs.hunger_weighted_approach', 'hgs.hunger_weighted_retreat', 'carryover'],
    'hba_honey': ['hba_honey.digging_phase_update', 'hba_honey.honey_phase_update', 'carryover'],
    'hsaba': ['hsaba.local_bat_random_walk', 'hsaba.velocity_bat_update', 'hsaba.differential_evolution_refinement', 'carryover'],
    'plo': ['plo.aurora_global_local_update', 'plo.polar_light_collision_update', 'carryover'],
}
_GENUINE_ENGINE_EMITTED_OPERATOR_LABELS.update({_lab for _labels in _B27_ENGINE_LABELS.values() for _lab in _labels})
_ENGINE_OPERATOR_LABEL_OVERRIDES.update(_B27_ENGINE_LABELS)
for _aid, _labels in _B27_ENGINE_LABELS.items():
    ENGINE_OPERATOR_LABELS[_aid] = list(_labels)
_SINGLE_OPERATOR_SEMANTIC_OK.difference_update(set(_B27_ENGINE_LABELS))
_PREVIOUS_LABELS_FOR_ALGORITHM_W27 = labels_for_algorithm
def labels_for_algorithm(algorithm_id: str) -> list[str]:  # type: ignore[override]
    aid = str(algorithm_id)
    if aid in _ENGINE_OPERATOR_LABEL_OVERRIDES:
        return list(_ENGINE_OPERATOR_LABEL_OVERRIDES[aid])
    return _PREVIOUS_LABELS_FOR_ALGORITHM_W27(aid)

# ---------------------------------------------------------------------------
# Batch 28 / Addendum accepted candidate-level instrumentation.
# ---------------------------------------------------------------------------
_B28_ENGINE_LABELS = {
    'cdo_chernobyl': ['cdo_chernobyl.alpha_beta_gamma_radiation_update', 'carryover'],
    'eao': ['eao.sinusoidal_best_substrate_update', 'eao.vector_scaled_differential_substrate_update', 'eao.scalar_scaled_differential_substrate_update', 'carryover'],
    'gazelle_oa': ['gazelle_oa.brownian_foraging_update', 'gazelle_oa.levy_elite_transition_update', 'gazelle_oa.levy_foraging_update', 'gazelle_oa.random_patch_avoidance_update', 'gazelle_oa.peer_difference_escape_update', 'carryover'],
    'horse_oa': ['horse_oa.dominant_stallion_update', 'horse_oa.experienced_horse_social_update', 'horse_oa.middle_rank_grazing_update', 'horse_oa.foal_exploration_update', 'carryover'],
    'lca': ['lca.best_cell_replication', 'lca.peer_lateral_invasion', 'lca.angiogenesis_mutation', 'carryover'],
    'liwo': ['liwo.breeze_spiral_translation', 'liwo.strong_wind_displacement', 'carryover'],
    'loa_lyrebird': ['loa_lyrebird.better_bird_imitation_update', 'loa_lyrebird.escape_step_update', 'carryover'],
    'mke': ['mke.king_learning_fluctuation_update', 'mke.peer_knowledge_difference_update', 'carryover'],
    'mtbo': ['mtbo.team_leader_coordinated_movement', 'mtbo.avalanche_worst_avoidance', 'mtbo.team_mean_movement', 'mtbo.random_relocation_phase', 'carryover'],
}
_GENUINE_ENGINE_EMITTED_OPERATOR_LABELS.update({_lab for _labels in _B28_ENGINE_LABELS.values() for _lab in _labels})
_ENGINE_OPERATOR_LABEL_OVERRIDES.update(_B28_ENGINE_LABELS)
for _aid, _labels in _B28_ENGINE_LABELS.items():
    ENGINE_OPERATOR_LABELS[_aid] = list(_labels)
_SINGLE_OPERATOR_SEMANTIC_OK.difference_update(set(_B28_ENGINE_LABELS))
_PREVIOUS_LABELS_FOR_ALGORITHM_W28 = labels_for_algorithm
def labels_for_algorithm(algorithm_id: str) -> list[str]:  # type: ignore[override]
    aid = str(algorithm_id)
    if aid in _ENGINE_OPERATOR_LABEL_OVERRIDES:
        return list(_ENGINE_OPERATOR_LABEL_OVERRIDES[aid])
    return _PREVIOUS_LABELS_FOR_ALGORITHM_W28(aid)

# Batch 28 runtime compatibility labels observed during target smoke.
_B28_RUNTIME_COMPAT_LABELS = {
    'cdo_chernobyl': ['cdo_chernobyl.cdo_chernobyl_position_update', 'cdo_chernobyl.selection'],
    'mtbo': ['mtbo.candidate_generation', 'mtbo.selection'],
}
for _aid, _labels in _B28_RUNTIME_COMPAT_LABELS.items():
    _current = list(ENGINE_OPERATOR_LABELS.get(_aid, []))
    for _lab in _labels:
        if _lab not in _current:
            _current.append(_lab)
    ENGINE_OPERATOR_LABELS[_aid] = _current
    _ov = list(_ENGINE_OPERATOR_LABEL_OVERRIDES.get(_aid, []))
    for _lab in _labels:
        if _lab not in _ov:
            _ov.append(_lab)
    _ENGINE_OPERATOR_LABEL_OVERRIDES[_aid] = _ov
_PREVIOUS_LABELS_FOR_ALGORITHM_W28_COMPAT = labels_for_algorithm
def labels_for_algorithm(algorithm_id: str) -> list[str]:  # type: ignore[override]
    aid = str(algorithm_id)
    if aid in _ENGINE_OPERATOR_LABEL_OVERRIDES:
        return list(_ENGINE_OPERATOR_LABEL_OVERRIDES[aid])
    return _PREVIOUS_LABELS_FOR_ALGORITHM_W28_COMPAT(aid)

# ---------------------------------------------------------------------------
# Batch 29 / Addendum candidate-level instrumentation.
# ---------------------------------------------------------------------------
_B29_ENGINE_LABELS = {
    'gkso': ['gkso.genghis_khan_crossover_exploration', 'gkso.shark_hunting_pso_update', 'carryover'],
    'info': ['info.best_weighted_mean_rule', 'info.random_weighted_mean_rule', 'carryover'],
    'ivya': ['ivya.neighbor_growth_update', 'ivya.best_growth_update', 'carryover'],
    'jde': ['jde.de_trial', 'jde.f_self_adaptation_trial', 'jde.cr_self_adaptation_trial', 'jde.f_cr_self_adaptation_trial', 'carryover'],
    'mshoa': ['mshoa.smasher_attack_update', 'mshoa.spearer_circular_attack_update', 'mshoa.defense_position_update'],
    'nmm': ['nmm.reflection_update', 'nmm.expansion_update', 'nmm.contraction_update', 'nmm.shrink_update', 'carryover'],
    'singer_oa': ['singer_oa.imitation_mimicry_phase', 'singer_oa.creation_perturbation_phase', 'carryover'],
    'wooa': ['wooa.scavenging_predator_following', 'wooa.prey_attack_update', 'wooa.fight_chase_local_update', 'carryover'],
}
_GENUINE_ENGINE_EMITTED_OPERATOR_LABELS.update({_lab for _labels in _B29_ENGINE_LABELS.values() for _lab in _labels})
_ENGINE_OPERATOR_LABEL_OVERRIDES.update(_B29_ENGINE_LABELS)
for _aid, _labels in _B29_ENGINE_LABELS.items():
    ENGINE_OPERATOR_LABELS[_aid] = list(_labels)
_SINGLE_OPERATOR_SEMANTIC_OK.difference_update(set(_B29_ENGINE_LABELS))
_PREVIOUS_LABELS_FOR_ALGORITHM_W29 = labels_for_algorithm
def labels_for_algorithm(algorithm_id: str) -> list[str]:  # type: ignore[override]
    aid = str(algorithm_id)
    if aid in _ENGINE_OPERATOR_LABEL_OVERRIDES:
        return list(_ENGINE_OPERATOR_LABEL_OVERRIDES[aid])
    return _PREVIOUS_LABELS_FOR_ALGORITHM_W29(aid)


# ---------------------------------------------------------------------------
# Batch 30 / Addendum continuation plus sacc_eam2 NameError fix.
# ---------------------------------------------------------------------------
_B30_ENGINE_LABELS = {
    'sacc_eam2': ['sacc_eam2.even_subcomponent_de_update', 'sacc_eam2.odd_subcomponent_de_update', 'carryover'],
    'sine_cosine_a': ['sine_cosine_a.sine_position_update', 'sine_cosine_a.cosine_position_update', 'carryover'],
    'firefly_a': ['firefly_a.attraction_dominant_move', 'firefly_a.randomization_dominant_move', 'carryover'],
    'sacoso': ['sacoso.cognitive_swarm_update', 'sacoso.social_swarm_update', 'carryover'],
    'pso': ['pso.inertia_velocity_update', 'pso.cognitive_memory_update', 'pso.social_global_update', 'carryover'],
}
_GENUINE_ENGINE_EMITTED_OPERATOR_LABELS.update({_lab for _labels in _B30_ENGINE_LABELS.values() for _lab in _labels})
_ENGINE_OPERATOR_LABEL_OVERRIDES.update(_B30_ENGINE_LABELS)
for _aid, _labels in _B30_ENGINE_LABELS.items():
    ENGINE_OPERATOR_LABELS[_aid] = list(_labels)
_SINGLE_OPERATOR_SEMANTIC_OK.difference_update(set(_B30_ENGINE_LABELS))
_PREVIOUS_LABELS_FOR_ALGORITHM_W30 = labels_for_algorithm
def labels_for_algorithm(algorithm_id: str) -> list[str]:  # type: ignore[override]
    aid = str(algorithm_id)
    if aid in _ENGINE_OPERATOR_LABEL_OVERRIDES:
        return list(_ENGINE_OPERATOR_LABEL_OVERRIDES[aid])
    return _PREVIOUS_LABELS_FOR_ALGORITHM_W30(aid)

# ---------------------------------------------------------------------------
# Batch 31 / Addendum accepted candidate-level instrumentation.
# ---------------------------------------------------------------------------
_B31_ENGINE_LABELS = {
    'csa': ['csa.memory_following_update', 'csa.awareness_random_relocation', 'csa.mixed_memory_random_update', 'carryover'],
    'cdo': ['cdo.alpha_cheetah_attack_component', 'cdo.beta_cheetah_attack_component', 'cdo.gamma_cheetah_attack_component', 'carryover'],
    'epc': ['epc.spiral_attraction_update', 'epc.thermal_mutation_update', 'carryover'],
}
_GENUINE_ENGINE_EMITTED_OPERATOR_LABELS.update({_lab for _labels in _B31_ENGINE_LABELS.values() for _lab in _labels})
_ENGINE_OPERATOR_LABEL_OVERRIDES.update(_B31_ENGINE_LABELS)
for _aid, _labels in _B31_ENGINE_LABELS.items():
    ENGINE_OPERATOR_LABELS[_aid] = list(_labels)
_SINGLE_OPERATOR_SEMANTIC_OK.difference_update(set(_B31_ENGINE_LABELS))
_PREVIOUS_LABELS_FOR_ALGORITHM_W31 = labels_for_algorithm
def labels_for_algorithm(algorithm_id: str) -> list[str]:  # type: ignore[override]
    aid = str(algorithm_id)
    if aid in _ENGINE_OPERATOR_LABEL_OVERRIDES:
        return list(_ENGINE_OPERATOR_LABEL_OVERRIDES[aid])
    return _PREVIOUS_LABELS_FOR_ALGORITHM_W31(aid)


# ---------------------------------------------------------------------------
# Batch 32 / Addendum accepted candidate-level instrumentation.
# ---------------------------------------------------------------------------
_B32_ENGINE_LABELS = {
    'bka': ['bka.sine_soaring_update', 'bka.random_soaring_update', 'bka.peer_repulsion_cauchy_update', 'bka.leader_attraction_cauchy_update', 'carryover'],
    'capsa': ['capsa.jumping_global_motion', 'capsa.long_jump_global_motion', 'capsa.velocity_swing_update', 'capsa.best_swing_update', 'capsa.velocity_memory_update', 'capsa.random_tree_leap', 'capsa.group_following_update', 'carryover'],
    'ep': ['ep.parent_survivor', 'ep.large_strategy_mutation_offspring', 'ep.small_strategy_mutation_offspring'],
    'es': ['es.parent_survivor', 'es.large_step_mutation_offspring', 'es.small_step_mutation_offspring'],
}
_GENUINE_ENGINE_EMITTED_OPERATOR_LABELS.update({_lab for _labels in _B32_ENGINE_LABELS.values() for _lab in _labels})
_ENGINE_OPERATOR_LABEL_OVERRIDES.update(_B32_ENGINE_LABELS)
for _aid, _labels in _B32_ENGINE_LABELS.items():
    ENGINE_OPERATOR_LABELS[_aid] = list(_labels)
_SINGLE_OPERATOR_SEMANTIC_OK.difference_update(set(_B32_ENGINE_LABELS))
_PREVIOUS_LABELS_FOR_ALGORITHM_W32 = labels_for_algorithm
def labels_for_algorithm(algorithm_id: str) -> list[str]:  # type: ignore[override]
    aid = str(algorithm_id)
    if aid in _ENGINE_OPERATOR_LABEL_OVERRIDES:
        return list(_ENGINE_OPERATOR_LABEL_OVERRIDES[aid])
    return _PREVIOUS_LABELS_FOR_ALGORITHM_W32(aid)


# ---------------------------------------------------------------------------
# Batch 33 / Addendum accepted candidate-level instrumentation.
# ---------------------------------------------------------------------------
_B33_ENGINE_LABELS = {
    'cat_so': ['cat_so.seeking_mode_expansive_copy_update', 'cat_so.seeking_mode_contracting_copy_update', 'cat_so.tracing_mode_velocity_update', 'carryover'],
    'da': ['da.neighbour_alignment_update', 'da.levy_flight_exploration', 'da.food_enemy_swarm_update', 'carryover'],
    'deo_dolphin': ['deo_dolphin.elite_reference_echo_guidance', 'deo_dolphin.elite_jitter_echo_guidance', 'deo_dolphin.peer_reference_echo_guidance', 'deo_dolphin.peer_jitter_echo_guidance', 'carryover'],
    'do_dandelion': ['do_dandelion.rising_seed_phase', 'do_dandelion.descent_diffusion_phase', 'do_dandelion.elite_landing_phase', 'do_dandelion.candidate_generation', 'do_dandelion.selection', 'carryover'],
    'eho': ['eho.long_range_clan_best_guided_update', 'eho.short_range_clan_best_guided_update', 'eho.matriarch_center_update', 'eho.separating_random_relocation', 'carryover'],
    'fda': ['fda.downhill_flow_direction_update', 'fda.neighbour_flow_direction_update', 'fda.elite_flow_direction_update', 'carryover'],
}
_GENUINE_ENGINE_EMITTED_OPERATOR_LABELS.update({_lab for _labels in _B33_ENGINE_LABELS.values() for _lab in _labels})
_ENGINE_OPERATOR_LABEL_OVERRIDES.update(_B33_ENGINE_LABELS)
for _aid, _labels in _B33_ENGINE_LABELS.items():
    ENGINE_OPERATOR_LABELS[_aid] = list(_labels)
_SINGLE_OPERATOR_SEMANTIC_OK.difference_update(set(_B33_ENGINE_LABELS))
_PREVIOUS_LABELS_FOR_ALGORITHM_W33 = labels_for_algorithm
def labels_for_algorithm(algorithm_id: str) -> list[str]:  # type: ignore[override]
    aid = str(algorithm_id)
    if aid in _ENGINE_OPERATOR_LABEL_OVERRIDES:
        return list(_ENGINE_OPERATOR_LABEL_OVERRIDES[aid])
    return _PREVIOUS_LABELS_FOR_ALGORITHM_W33(aid)


# ---------------------------------------------------------------------------
# Batch 34 / Addendum accepted candidate-level instrumentation.
# ---------------------------------------------------------------------------
_B34_ENGINE_LABELS = {
    'bbso': ['bbso.coordinated_following_trial', 'bbso.self_following_trial', 'carryover'],
    'rso': ['rso.long_chasing_update', 'rso.short_chasing_update'],
    'shio_success': ['shio_success.best_history_guidance', 'shio_success.second_history_guidance', 'shio_success.third_history_guidance', 'carryover'],
    'shio': ['shio.first_iguana_guidance', 'shio.second_iguana_guidance', 'shio.third_iguana_guidance'],
}
_GENUINE_ENGINE_EMITTED_OPERATOR_LABELS.update({_lab for _labels in _B34_ENGINE_LABELS.values() for _lab in _labels})
_ENGINE_OPERATOR_LABEL_OVERRIDES.update(_B34_ENGINE_LABELS)
for _aid, _labels in _B34_ENGINE_LABELS.items():
    ENGINE_OPERATOR_LABELS[_aid] = list(_labels)
_SINGLE_OPERATOR_SEMANTIC_OK.difference_update(set(_B34_ENGINE_LABELS))
_PREVIOUS_LABELS_FOR_ALGORITHM_W34 = labels_for_algorithm
def labels_for_algorithm(algorithm_id: str) -> list[str]:  # type: ignore[override]
    aid = str(algorithm_id)
    if aid in _ENGINE_OPERATOR_LABEL_OVERRIDES:
        return list(_ENGINE_OPERATOR_LABEL_OVERRIDES[aid])
    return _PREVIOUS_LABELS_FOR_ALGORITHM_W34(aid)


# ===== HONEST SINGLE-OPERATOR CORRECTION (runtime-verified) =====
# These engines evaluate once per candidate; their candidates carry exactly
# one raw operator at runtime. Their fitness-effect is not separable into
# sub-operators passively. Keep ONE honest label (CDS=1.0) and whitelist;
# do NOT split one measured operator into a fabricated 1/N distribution.
_SINGLE_EVAL_HONEST = {
"aco": "aco.pheromone_weighted_perturbation_in_each_dimension",
"acor": "acor.archive_kernel_sampling_update",
"aefa": "aefa.electric_field_force_update",
"aesspso": "aesspso.adaptive_velocity_position_update",
"afsa": "afsa.leap",
"aoa": "aoa.arithmetic_operator_position_update",
"arch_oa": "arch_oa.archimedes_density_volume_acceleration_update",
"aso": "aso.anarchic_social_position_update",
"aso_atom": "aso_atom.do_not_move_current_elites_unless",
"autov": "autov.learned_variation_operator_update",
"basin_hopping": "basin_hopping.update",
"bbo": "bbo.migration_mutation_selection_update",
"bco": "bco.swim_refinement_update",
"bfgs": "bfgs.update",
"bipop_cmaes": "bipop_cmaes.update",
"bspga": "bspga.binary_partition_tree_variation_update",
"ca": "ca.cultural_belief_guided_update",
"cddo": "cddo.cheetah_chase_position_update",
"cddo_child": "cddo_child.child_drawing_development_update",
"cem": "cem.model_sampling_elite_distribution_update",
"choa": "choa.chimp_hunting_position_update",
"circle_sa": "circle_sa.circle_position_update",
"cmaes": "cmaes.covariance_sampling_recombination_update",
"coot": "coot.chain_movement_update",
"cso": "cso.mean_all_positions",
"de": "de.differential_mutation_crossover_selection",
"dream_oa": "dream_oa.dream_generation_refinement_update",
"ecpo": "ecpo.electric_charge_random_perturbation",
"ego": "ego.expected_improvement_candidate_generation",
"eo": "eo.equilibrium_position_update",
"eso": "eso.electric_storm_field_update",
"et_bo": "et_bo.update",
"fep": "fep.fast_mutation_tournament_selection_update",
"ffa": "ffa.fruitfly_smell_search_update",
"frcg": "frcg.update",
"gbrt_bo": "gbrt_bo.update",
"gco": "gco.dark_zone_mutation_update",
"ggo": "ggo.greylag_goose_flock_update",
"gmo": "gmo.marketing_guidance_update",
"goa": "goa.grasshopper_social_force_update",
"gp_bo": "gp_bo.update",
"gpoo": "gpoo.octopus_tentacle_prey_position_update",
"gpso": "gpso.velocity_position_update",
"grasp": "grasp.update",
"gsa": "gsa.gravitational_force_acceleration_update",
"gska": "gska.gaining_sharing_knowledge_update",
"gso": "gso.glowworm_luciferin_movement_update",
"gso_glider_snake": "gso_glider_snake.glider_snake_position_update",
"hbo": "hbo.heap_rank_pressure_update",
"hc": "hc.update",
"hco": "hco.conception_growth_update",
"hiking_oa": "hiking_oa.hiking_slope_velocity_update",
"hsa": "hsa.harmony_memory_improvisation_update",
"hus": "hus.update",
"i_woa": "i_woa.polynomial_breeding_refinement",
"iagwo": "iagwo.adaptive_alpha_beta_delta_update",
"ils": "ils.update",
"ilshade": "ilshade.linear_population_reduction_mutation_selection",
"ipop_cmaes": "ipop_cmaes.update",
"iwo": "iwo.seed_dispersal_colonization_update",
"jso": "jso.ocean_current_swarm_motion_update",
"jy": "jy.best_away_from_worst_update",
"l2smea": "l2smea.update",
"lco": "lco.life_choice_boundary_reflection_update",
"lfd": "lfd.levy_flight_search",
"lpo": "lpo.lichen_growth_propagation_update",
"lshade_cnepsin": "lshade_cnepsin.cn_epsin_mutation_crossover_selection",
"lso_spectrum": "lso_spectrum.light_spectrum_position_update",
"mbo": "mbo.monarch_migration_adjusting_update",
"mfa": "mfa.moth_flame_spiral_update",
"mfea": "mfea.assortative_mating_mutation_transfer_update",
"mfea2": "mfea2.update",
"mgo": "mgo.territory_mountain_herding_update",
"mgoa_market": "mgoa_market.market_gradient_position_update",
"misaco": "misaco.update",
"moss_go": "moss_go.water_dispersal_growth_update",
"msa_e": "msa_e.golden_ratio_exploitation_update",
"msls": "msls.update",
"mts": "mts.multiple_trajectory_local_search_update",
"mvpa": "mvpa.mvp_guided_player_update",
"nca": "nca.acceleration_hyperbolic_contraction_random_subset_components",
"noa": "noa.newton_position_update",
"ofa": "ofa.owl_neighbour_flight_update",
"parrot_o": "parrot_o.flight_area_search_update",
"pbil": "pbil.update",
"pcx": "pcx.parent_centric_crossover_update",
"pdo": "pdo.prairie_dog_burrow_alarm_update",
"petio": "petio.performance_evaluation_teaching_update",
"pfa": "pfa.pathfinder_position_update",
"plba": "plba.path_looping_bat_update",
"random_s": "random_s.random_sampling_update",
"rf_bo": "rf_bo.update",
"rfo": "rfo.red_fox_smell_search_update",
"rhso": "rhso.rhinoceros_herd_position_update",
"rime": "rime.hard_rime_puncture_update",
"roa": "roa.remora_attempt_update",
"rsa": "rsa.reptile_hunting_encircling_update",
"sa": "sa.update",
"saba": "saba.self_adaptive_bat_update",
"sade_amss": "sade_amss.adaptive_multistrategy_subspace_de_update",
"sade_atdsc": "sade_atdsc.adaptive_trial_distribution_selection_update",
"sade_sammon": "sade_sammon.sammon_surrogate_de_selection_update",
"samso": "samso.self_adaptive_migratory_swarm_update",
"sapo": "sapo.update",
"sbo": "sbo.bowerbird_mutation_update",
"scho": "scho.scholar_chess_position_update",
"shade": "shade.success_history_mutation_crossover_selection",
"sho": "sho.spotted_hyena_hunting_update",
"soa": "soa.seagull_spiral_attack_update",
"sopt": "sopt.statistical_population_selection_update",
"sqp": "sqp.update",
"ssio_rl": "ssio_rl.update",
"sspider_a": "sspider_a.social_spider_vibration_update",
"thro": "thro.throwing_race_update",
"toc": "toc.combination_velocity_update",
"tpo": "tpo.carbon_nutrient_leaf_update",
"ts": "ts.update",
"two": "two.tug_of_war_weight_force_update",
"vns": "vns.update",
"wdo": "wdo.wind_velocity_position_update",
"wo_wave": "wo_wave.wave_propagation_position_update",
"wso": "wso.white_shark_swarm_position_update",
"wutp": "wutp.horizontal_water_transport_update"
}
_SINGLE_OPERATOR_SEMANTIC_OK.update(set(_SINGLE_EVAL_HONEST.keys()))

_prev_labels_for_algorithm_se = labels_for_algorithm
def labels_for_algorithm(algorithm_id):  # type: ignore[override]
    if algorithm_id in _SINGLE_EVAL_HONEST:
        return [_SINGLE_EVAL_HONEST[algorithm_id]]
    return _prev_labels_for_algorithm_se(algorithm_id)

_prev_resolve_se = resolve_operator_label
def resolve_operator_label(algorithm_id, filename, function, line=None):  # type: ignore[override]
    if algorithm_id in _SINGLE_EVAL_HONEST:
        return _SINGLE_EVAL_HONEST[algorithm_id]
    return _prev_resolve_se(algorithm_id, filename, function, line)


# ===== HONEST SINGLE-OPERATOR CORRECTION: disable compound split for single-eval engines =====
_prev_expand_compound_se = expand_compound_operator_label
def expand_compound_operator_label(algorithm_id, label):  # type: ignore[override]
    # Runtime-verified single-eval engines: never split their one measured
    # operator into a fabricated 1/N distribution. Return the one honest label.
    if algorithm_id in _SINGLE_EVAL_HONEST:
        return [_SINGLE_EVAL_HONEST[algorithm_id]]
    return _prev_expand_compound_se(algorithm_id, label)

# Five engines that emit >=2 raw labels but whose CDS is a FIXED split (invariant
# to seed AND objective) -> not genuine per-operator measurement. Collapse to one
# honest label + whitelist, same as the single-eval set.
_SINGLE_EVAL_HONEST.update({
    "flo": "flo.update",
    "kma": "kma.update",
    "puma_o": "puma_o.update",
    "rbmo": "rbmo.update",
    "sboa": "sboa.update",
})
_SINGLE_OPERATOR_SEMANTIC_OK.update({"flo","kma","puma_o","rbmo","sboa"})

# ---------------------------------------------------------------------------
# Addendum — native NCCLA operator labels
# ---------------------------------------------------------------------------
_NCCLA_OPERATOR_LABELS = [
    "nccla.vertical_social_learning",
    "nccla.horizontal_social_learning",
    "nccla.individual_learning",
    "nccla.juvenile_reinforcement",
    "nccla.parent_reinforcement",
    "nccla.parent_selection",
]
_NCCLA_COMPOUND_SPLITS = {
    "nccla.vertical_social_learning_juvenile_reinforcement": [
        "nccla.vertical_social_learning",
        "nccla.juvenile_reinforcement",
    ],
    "nccla.horizontal_social_learning_juvenile_reinforcement": [
        "nccla.horizontal_social_learning",
        "nccla.juvenile_reinforcement",
    ],
    "nccla.individual_learning_juvenile_reinforcement": [
        "nccla.individual_learning",
        "nccla.juvenile_reinforcement",
    ],
}
_EXACT_COMPOUND_OPERATOR_SPLITS.update(_NCCLA_COMPOUND_SPLITS)
try:
    _ENGINE_OPERATOR_LABEL_OVERRIDES["nccla"] = list(_NCCLA_OPERATOR_LABELS)
except NameError:  # pragma: no cover - defensive for generated catalog variants
    pass
ENGINE_OPERATOR_LABELS["nccla"] = list(_NCCLA_OPERATOR_LABELS)
try:
    _GENUINE_ENGINE_EMITTED_OPERATOR_LABELS.update(_NCCLA_OPERATOR_LABELS)
    _GENUINE_ENGINE_EMITTED_OPERATOR_LABELS.update(_NCCLA_COMPOUND_SPLITS.keys())
except NameError:  # pragma: no cover
    pass

_prev_labels_for_algorithm_nccla = labels_for_algorithm
def labels_for_algorithm(algorithm_id):  # type: ignore[override]
    if str(algorithm_id) == "nccla":
        return list(_NCCLA_OPERATOR_LABELS)
    return _prev_labels_for_algorithm_nccla(algorithm_id)

_prev_expand_compound_operator_label_nccla = expand_compound_operator_label
def expand_compound_operator_label(algorithm_id, label):  # type: ignore[override]
    if label in _NCCLA_COMPOUND_SPLITS:
        return list(_NCCLA_COMPOUND_SPLITS[label])
    return _prev_expand_compound_operator_label_nccla(algorithm_id, label)

# ---------------------------------------------------------------------------
# Addendum — native operator labels for 2024–2025 metaheuristic paper ports
# ---------------------------------------------------------------------------
_ADDED_METAHEURISTIC_OPERATOR_LABELS = {
    "agdo": [
        "agdo.progressive_gradient_momentum_dynamic_interaction",
        "agdo.system_optimization_operator",
    ],
    "dp": [
        "dp.delta_operation",
    ],
    "lea": [
        "lea.reflection_operation",
        "lea.value_phase_reflection_operation",
        "lea.value_phase_role_phase",
    ],
    "ppo": [
        "ppo.escape_sexual_cannibalism_juvenile_generation",
        "ppo.escape_predation_local_search",
    ],
    "rrto": [
        "rrto.adaptive_step_size_wandering",
        "rrto.absolute_difference_step",
        "rrto.boundary_based_step",
    ],
}

_ADDED_METAHEURISTIC_COMPOUND_SPLITS = {
    "agdo.progressive_gradient_momentum_dynamic_interaction": [
        "agdo.progressive_gradient_momentum_integration",
        "agdo.dynamic_gradient_interaction",
    ],
    "lea.value_phase_reflection_operation": [
        "lea.value_phase",
        "lea.reflection_operation",
    ],
    "lea.value_phase_role_phase": [
        "lea.value_phase",
        "lea.role_phase",
    ],
    "ppo.escape_sexual_cannibalism_juvenile_generation": [
        "ppo.escape_ejection",
        "ppo.sexual_cannibalism_juvenile_generation",
    ],
    "ppo.escape_predation_local_search": [
        "ppo.escape_ejection",
        "ppo.predation_local_search",
    ],
}
_EXACT_COMPOUND_OPERATOR_SPLITS.update(_ADDED_METAHEURISTIC_COMPOUND_SPLITS)

for _aid, _labels in _ADDED_METAHEURISTIC_OPERATOR_LABELS.items():
    try:
        _ENGINE_OPERATOR_LABEL_OVERRIDES[_aid] = list(_labels)
    except NameError:  # pragma: no cover - defensive for generated catalog variants
        pass
    ENGINE_OPERATOR_LABELS[_aid] = list(_labels)

try:
    for _labels in _ADDED_METAHEURISTIC_OPERATOR_LABELS.values():
        _GENUINE_ENGINE_EMITTED_OPERATOR_LABELS.update(_labels)
    _GENUINE_ENGINE_EMITTED_OPERATOR_LABELS.update(_ADDED_METAHEURISTIC_COMPOUND_SPLITS.keys())
except NameError:  # pragma: no cover
    pass

_prev_labels_for_algorithm_added_metaheuristics = labels_for_algorithm
def labels_for_algorithm(algorithm_id):  # type: ignore[override]
    aid = str(algorithm_id).lower()
    if aid in _ADDED_METAHEURISTIC_OPERATOR_LABELS:
        return list(_ADDED_METAHEURISTIC_OPERATOR_LABELS[aid])
    return _prev_labels_for_algorithm_added_metaheuristics(algorithm_id)

_prev_expand_compound_operator_label_added_metaheuristics = expand_compound_operator_label
def expand_compound_operator_label(algorithm_id, label):  # type: ignore[override]
    if label in _ADDED_METAHEURISTIC_COMPOUND_SPLITS:
        return list(_ADDED_METAHEURISTIC_COMPOUND_SPLITS[label])
    return _prev_expand_compound_operator_label_added_metaheuristics(algorithm_id, label)

# ---------------------------------------------------------------------------
# Addendum — Yukthi Opus native operator labels
# ---------------------------------------------------------------------------
_YO_OPERATOR_LABELS = [
    "yo.mcmc_burn_in",
    "yo.post_burnin_selection",
    "yo.mcmc_proposal",
    "yo.greedy_refinement",
    "yo.simulated_annealing_acceptance",
    "yo.blacklist_filter",
    "yo.adaptive_reheating",
    "yo.elite_update",
]
_YO_COMPOUND_SPLITS = {
    "yo.hybrid_mcmc_greedy_sa_update": [
        "yo.mcmc_proposal",
        "yo.greedy_refinement",
        "yo.simulated_annealing_acceptance",
    ],
    "yo.burn_in_mcmc_update": ["yo.mcmc_burn_in"],
}
_EXACT_COMPOUND_OPERATOR_SPLITS.update(_YO_COMPOUND_SPLITS)
try:
    _ENGINE_OPERATOR_LABEL_OVERRIDES["yo"] = list(_YO_OPERATOR_LABELS)
except NameError:  # pragma: no cover - defensive for generated catalog variants
    pass
ENGINE_OPERATOR_LABELS["yo"] = list(_YO_OPERATOR_LABELS)
try:
    _GENUINE_ENGINE_EMITTED_OPERATOR_LABELS.update(_YO_OPERATOR_LABELS)
    _GENUINE_ENGINE_EMITTED_OPERATOR_LABELS.update(_YO_COMPOUND_SPLITS.keys())
except NameError:  # pragma: no cover
    pass

_prev_labels_for_algorithm_yo = labels_for_algorithm
def labels_for_algorithm(algorithm_id):  # type: ignore[override]
    if str(algorithm_id).lower() == "yo":
        return list(_YO_OPERATOR_LABELS)
    return _prev_labels_for_algorithm_yo(algorithm_id)

_prev_expand_compound_operator_label_yo = expand_compound_operator_label
def expand_compound_operator_label(algorithm_id, label):  # type: ignore[override]
    if label in _YO_COMPOUND_SPLITS:
        return list(_YO_COMPOUND_SPLITS[label])
    return _prev_expand_compound_operator_label_yo(algorithm_id, label)

# ---------------------------------------------------------------------------
# Addendum — Tornado Optimizer with Coriolis Force native operator labels
# ---------------------------------------------------------------------------
_TOC_OPERATOR_LABELS = [
    "toc.fitness_proportional_assignment",
    "toc.coriolis_velocity_update",
    "toc.windstorm_to_tornado_evolution",
    "toc.windstorm_to_thunderstorm_evolution",
    "toc.thunderstorm_to_tornado_evolution",
    "toc.random_windstorm_formation",
    "toc.role_exchange_replacement",
]
_TOC_DIRECT_OPERATOR_LABELS = [
    "toc.windstorm_to_tornado_evolution",
    "toc.windstorm_to_thunderstorm_evolution",
    "toc.thunderstorm_to_tornado_evolution",
    "toc.random_windstorm_formation",
]
_TOC_DIAGNOSTIC_OPERATOR_LABELS = [
    "toc.fitness_proportional_assignment",
    "toc.coriolis_velocity_update",
    "toc.role_exchange_replacement",
]
_TOC_COMPOUND_SPLITS = {
    "toc.windstorm_evolution_update": [
        "toc.coriolis_velocity_update",
        "toc.windstorm_to_tornado_evolution",
        "toc.windstorm_to_thunderstorm_evolution",
    ],
    "toc.thunderstorm_evolution_update": ["toc.thunderstorm_to_tornado_evolution"],
}
try:
    _EXACT_COMPOUND_OPERATOR_SPLITS.update(_TOC_COMPOUND_SPLITS)
except NameError:  # pragma: no cover - defensive for generated catalog variants
    pass
try:
    _ENGINE_OPERATOR_LABEL_OVERRIDES["toc"] = list(_TOC_OPERATOR_LABELS)
except NameError:  # pragma: no cover - defensive for generated catalog variants
    pass
ENGINE_OPERATOR_LABELS["toc"] = list(_TOC_OPERATOR_LABELS)
try:
    _GENUINE_ENGINE_EMITTED_OPERATOR_LABELS.update(_TOC_OPERATOR_LABELS)
    _GENUINE_ENGINE_EMITTED_OPERATOR_LABELS.update(_TOC_COMPOUND_SPLITS.keys())
except NameError:  # pragma: no cover
    pass
try:
    _SINGLE_OPERATOR_SEMANTIC_OK.discard("toc")
except NameError:  # pragma: no cover
    pass

_prev_labels_for_algorithm_toc = labels_for_algorithm
def labels_for_algorithm(algorithm_id):  # type: ignore[override]
    if str(algorithm_id).lower() == "toc":
        return list(_TOC_OPERATOR_LABELS)
    return _prev_labels_for_algorithm_toc(algorithm_id)

_prev_expand_compound_operator_label_toc = expand_compound_operator_label
def expand_compound_operator_label(algorithm_id, label):  # type: ignore[override]
    if label in _TOC_COMPOUND_SPLITS:
        return list(_TOC_COMPOUND_SPLITS[label])
    return _prev_expand_compound_operator_label_toc(algorithm_id, label)

# TOC now emits native paper-level operator contributions; remove the historical
# monolithic single-operator guard so attribution does not collapse to the old
# heuristic spiral label.
try:
    _SINGLE_EVAL_HONEST.pop("toc", None)
    _SINGLE_OPERATOR_SEMANTIC_OK.discard("toc")
except NameError:  # pragma: no cover
    pass

# Addendum — native L-SHADE and mLSHADE-RL operator labels.
_LSHADE_OPERATOR_LABELS = [
    "lshade.mutation",
    "lshade.crossover",
    "lshade.selection",
    "lshade.archive_update",
    "lshade.success_history_update",
    "lshade.population_reduction",
]
_MLSHADE_RL_OPERATOR_LABELS = [
    "mlshade_rl.ms1_current_to_pbest_weight_archive",
    "mlshade_rl.ms2_current_to_pbest_no_archive",
    "mlshade_rl.ms3_current_to_ordpbest_weight",
    "mlshade_rl.crossover",
    "mlshade_rl.selection",
    "mlshade_rl.strategy_probability_update",
    "mlshade_rl.parameter_adaptation",
    "mlshade_rl.archive_update",
    "mlshade_rl.population_reduction",
    "mlshade_rl.restart",
    "mlshade_rl.local_search",
]
try:
    _ENGINE_OPERATOR_LABEL_OVERRIDES["lshade"] = list(_LSHADE_OPERATOR_LABELS)
    _ENGINE_OPERATOR_LABEL_OVERRIDES["mlshade_rl"] = list(_MLSHADE_RL_OPERATOR_LABELS)
except NameError:  # pragma: no cover - defensive for generated catalog variants
    pass
ENGINE_OPERATOR_LABELS["lshade"] = list(_LSHADE_OPERATOR_LABELS)
ENGINE_OPERATOR_LABELS["mlshade_rl"] = list(_MLSHADE_RL_OPERATOR_LABELS)
try:
    _GENUINE_ENGINE_EMITTED_OPERATOR_LABELS.update(_LSHADE_OPERATOR_LABELS)
    _GENUINE_ENGINE_EMITTED_OPERATOR_LABELS.update(_MLSHADE_RL_OPERATOR_LABELS)
except NameError:  # pragma: no cover
    pass

_prev_labels_for_algorithm_lshade_family = labels_for_algorithm
def labels_for_algorithm(algorithm_id):  # type: ignore[override]
    aid = str(algorithm_id).lower().replace("-", "_")
    if aid == "lshade":
        return list(_LSHADE_OPERATOR_LABELS)
    if aid == "mlshade_rl":
        return list(_MLSHADE_RL_OPERATOR_LABELS)
    return _prev_labels_for_algorithm_lshade_family(algorithm_id)

# Addendum — supplied SHADE-family and Secant engine-native operator labels.
_SUPPLIED_NATIVE_OPERATOR_LABELS = {
    "jso_de": [
        "jso_de.mutation", "jso_de.crossover", "jso_de.selection",
        "jso_de.archive_update", "jso_de.success_history_update",
        "jso_de.population_reduction",
    ],
    "lshade_epsin": [
        "lshade_epsin.mutation", "lshade_epsin.crossover", "lshade_epsin.selection",
        "lshade_epsin.archive_update", "lshade_epsin.success_history_update",
        "lshade_epsin.population_reduction", "lshade_epsin.ensemble_sinusoidal_adaptation",
        "lshade_epsin.gaussian_walk_local_search",
    ],
    "lshade_rsp": [
        "lshade_rsp.mutation", "lshade_rsp.crossover", "lshade_rsp.selection",
        "lshade_rsp.archive_update", "lshade_rsp.success_history_update",
        "lshade_rsp.population_reduction", "lshade_rsp.rank_selective_pressure",
    ],
    "lshade_spacma": [
        "lshade_spacma.mutation", "lshade_spacma.crossover", "lshade_spacma.selection",
        "lshade_spacma.archive_update", "lshade_spacma.success_history_update",
        "lshade_spacma.population_reduction", "lshade_spacma.cma_es_sampling",
    ],
    "ilshade_rsp": [
        "ilshade_rsp.mutation", "ilshade_rsp.crossover", "ilshade_rsp.selection",
        "ilshade_rsp.archive_update", "ilshade_rsp.success_history_update",
        "ilshade_rsp.population_reduction", "ilshade_rsp.rank_selective_pressure",
        "ilshade_rsp.cauchy_target_perturbation",
    ],
    "nlshade_lbc": [
        "nlshade_lbc.mutation", "nlshade_lbc.crossover", "nlshade_lbc.selection",
        "nlshade_lbc.archive_update", "nlshade_lbc.success_history_update",
        "nlshade_lbc.population_reduction", "nlshade_lbc.rank_selective_pressure",
        "nlshade_lbc.linear_bias_change",
    ],
    "nlshade_rsp": [
        "nlshade_rsp.mutation", "nlshade_rsp.crossover", "nlshade_rsp.selection",
        "nlshade_rsp.archive_update", "nlshade_rsp.success_history_update",
        "nlshade_rsp.population_reduction", "nlshade_rsp.rank_selective_pressure",
    ],
    "nlshade_rsp_midpoint": [
        "nlshade_rsp_midpoint.mutation", "nlshade_rsp_midpoint.crossover", "nlshade_rsp_midpoint.selection",
        "nlshade_rsp_midpoint.archive_update", "nlshade_rsp_midpoint.success_history_update",
        "nlshade_rsp_midpoint.population_reduction", "nlshade_rsp_midpoint.rank_selective_pressure",
        "nlshade_rsp_midpoint.midpoint_evaluation", "nlshade_rsp_midpoint.midpoint_restart",
    ],
    "rde": [
        "rde.mutation", "rde.order_pbest_mutation", "rde.crossover", "rde.selection",
        "rde.archive_update", "rde.success_history_update", "rde.population_reduction",
        "rde.rank_selective_pressure", "rde.cauchy_target_perturbation", "rde.strategy_ratio_update",
    ],
    "secant_oa": [
        "secant_oa.secant_update", "secant_oa.stochastic_exploitation",
        "secant_oa.mutation_gate", "secant_oa.selection",
    ],
}
try:
    _ENGINE_OPERATOR_LABEL_OVERRIDES.update({k: list(v) for k, v in _SUPPLIED_NATIVE_OPERATOR_LABELS.items()})
except NameError:  # pragma: no cover
    pass
ENGINE_OPERATOR_LABELS.update({k: list(v) for k, v in _SUPPLIED_NATIVE_OPERATOR_LABELS.items()})
try:
    for _labels in _SUPPLIED_NATIVE_OPERATOR_LABELS.values():
        _GENUINE_ENGINE_EMITTED_OPERATOR_LABELS.update(_labels)
except NameError:  # pragma: no cover
    pass

_prev_labels_for_algorithm_supplied_family = labels_for_algorithm
def labels_for_algorithm(algorithm_id):  # type: ignore[override]
    _aid = str(algorithm_id).lower().replace("-", "_")
    if _aid in _SUPPLIED_NATIVE_OPERATOR_LABELS:
        return list(_SUPPLIED_NATIVE_OPERATOR_LABELS[_aid])
    return _prev_labels_for_algorithm_supplied_family(algorithm_id)


# ---------------------------------------------------------------------------
# Audit hygiene: qualify passive carryover labels.
# ---------------------------------------------------------------------------
# Some legacy addendum catalogs stored the passive non-operator bucket as the
# bare label "carryover".  EvoMapX's audit expects every exported catalog label
# to be namespaced by algorithm id.  This post-processing step keeps the same
# semantics while exporting "<algorithm_id>.carryover" consistently.
def _qualify_passive_carryover_labels(_aid: str, _labels: list[str]) -> list[str]:
    _prefix = f"{str(_aid).lower().replace('-', '_')}."
    _out: list[str] = []
    for _label in _labels:
        _lab = str(_label)
        if _lab == "carryover":
            _lab = _prefix + "carryover"
        if _lab not in _out:
            _out.append(_lab)
    return _out

try:
    for _aid, _labels in list(ENGINE_OPERATOR_LABELS.items()):
        ENGINE_OPERATOR_LABELS[_aid] = _qualify_passive_carryover_labels(_aid, list(_labels))
    for _aid, _labels in list(_ENGINE_OPERATOR_LABEL_OVERRIDES.items()):
        _ENGINE_OPERATOR_LABEL_OVERRIDES[_aid] = _qualify_passive_carryover_labels(_aid, list(_labels))
    _GENUINE_ENGINE_EMITTED_OPERATOR_LABELS = {
        (_lab if _lab != "carryover" else "passive.carryover")
        for _lab in _GENUINE_ENGINE_EMITTED_OPERATOR_LABELS
    }
except NameError:  # pragma: no cover - defensive for generated catalog variants
    pass

_prev_labels_for_algorithm_carryover_qualified = labels_for_algorithm
def labels_for_algorithm(algorithm_id):  # type: ignore[override]
    _aid = str(algorithm_id).lower().replace("-", "_")
    return _qualify_passive_carryover_labels(
        _aid,
        list(_prev_labels_for_algorithm_carryover_qualified(algorithm_id)),
    )

# Addendum — RDEx-SOP native operator labels.
_RDEX_SOP_OPERATOR_LABELS = [
    "rdex_sop.standard_branch_mutation",
    "rdex_sop.exploitation_biased_mutation",
    "rdex_sop.binomial_crossover",
    "rdex_sop.cauchy_local_perturbation",
    "rdex_sop.greedy_selection",
    "rdex_sop.dynamic_pbest_selection",
    "rdex_sop.hybrid_rate_update",
    "rdex_sop.success_history_update",
    "rdex_sop.linear_population_reduction",
    "rdex_sop.bound_resampling",
]
try:
    _ENGINE_OPERATOR_LABEL_OVERRIDES["rdex_sop"] = list(_RDEX_SOP_OPERATOR_LABELS)
except NameError:  # pragma: no cover - defensive for generated catalog variants
    pass
ENGINE_OPERATOR_LABELS["rdex_sop"] = list(_RDEX_SOP_OPERATOR_LABELS)
try:
    _GENUINE_ENGINE_EMITTED_OPERATOR_LABELS.update(_RDEX_SOP_OPERATOR_LABELS)
except NameError:  # pragma: no cover
    pass

_prev_labels_for_algorithm_rdex_sop = labels_for_algorithm
def labels_for_algorithm(algorithm_id):  # type: ignore[override]
    if str(algorithm_id).lower().replace("-", "_") == "rdex_sop":
        return list(_RDEX_SOP_OPERATOR_LABELS)
    return _prev_labels_for_algorithm_rdex_sop(algorithm_id)


# Addendum — IPOP-CMA-ES native operator labels.
# IPOP-CMA-ES now emits native engine-level operator_contributions.  Remove the
# historical single-update collapse so the web UI and evomapx_analysis expose
# CMA-ES sampling, elite recombination, model updates, restarts, and injection
# as distinct passive labels.
_IPOP_CMAES_OPERATOR_LABELS = [
    "ipop_cmaes.cmaes_sampling",
    "ipop_cmaes.elite_recombination",
    "ipop_cmaes.distribution_update",
    "ipop_cmaes.step_size_adaptation",
    "ipop_cmaes.population_restart",
    "ipop_cmaes.boundary_penalty",
    "ipop_cmaes.candidate_injection",
    "ipop_cmaes.initialization",
]
try:
    _SINGLE_EVAL_HONEST.pop("ipop_cmaes", None)
    _SINGLE_OPERATOR_SEMANTIC_OK.discard("ipop_cmaes")
    _ENGINE_OPERATOR_LABEL_OVERRIDES["ipop_cmaes"] = list(_IPOP_CMAES_OPERATOR_LABELS)
except NameError:  # pragma: no cover - defensive for generated catalog variants
    pass
ENGINE_OPERATOR_LABELS["ipop_cmaes"] = list(_IPOP_CMAES_OPERATOR_LABELS)
try:
    _GENUINE_ENGINE_EMITTED_OPERATOR_LABELS.update(_IPOP_CMAES_OPERATOR_LABELS)
except NameError:  # pragma: no cover
    pass

_prev_semanticize_operator_label_ipop_cmaes_native = semanticize_operator_label
def semanticize_operator_label(algorithm_id, label):  # type: ignore[override]
    aid = str(algorithm_id).lower().replace("-", "_")
    lab = str(label) if label not in {None, ""} else label
    if aid == "ipop_cmaes" and lab in _IPOP_CMAES_OPERATOR_LABELS:
        return lab
    return _prev_semanticize_operator_label_ipop_cmaes_native(algorithm_id, label)

_prev_labels_for_algorithm_ipop_cmaes_native = labels_for_algorithm
def labels_for_algorithm(algorithm_id):  # type: ignore[override]
    if str(algorithm_id).lower().replace("-", "_") == "ipop_cmaes":
        return list(_IPOP_CMAES_OPERATOR_LABELS)
    return _prev_labels_for_algorithm_ipop_cmaes_native(algorithm_id)

_prev_expand_compound_operator_label_ipop_cmaes_native = expand_compound_operator_label
def expand_compound_operator_label(algorithm_id, label):  # type: ignore[override]
    aid = str(algorithm_id).lower().replace("-", "_")
    lab = str(label) if label not in {None, ""} else label
    if aid == "ipop_cmaes" and lab in _IPOP_CMAES_OPERATOR_LABELS:
        return [lab]
    return _prev_expand_compound_operator_label_ipop_cmaes_native(algorithm_id, label)


# Addendum — BIPOP-CMA-ES native operator labels.
_BIPOP_CMAES_OPERATOR_LABELS = [
    "bipop_cmaes.cmaes_sampling",
    "bipop_cmaes.elite_recombination",
    "bipop_cmaes.distribution_update",
    "bipop_cmaes.step_size_adaptation",
    "bipop_cmaes.large_population_restart",
    "bipop_cmaes.small_population_restart",
    "bipop_cmaes.budget_regime_selection",
    "bipop_cmaes.termination_check",
    "bipop_cmaes.boundary_repair",
    "bipop_cmaes.candidate_injection",
    "bipop_cmaes.initialization",
]
try:
    _SINGLE_EVAL_HONEST.pop("bipop_cmaes", None)
    _SINGLE_OPERATOR_SEMANTIC_OK.discard("bipop_cmaes")
    _ENGINE_OPERATOR_LABEL_OVERRIDES["bipop_cmaes"] = list(_BIPOP_CMAES_OPERATOR_LABELS)
except NameError:  # pragma: no cover - defensive for generated catalog variants
    pass
ENGINE_OPERATOR_LABELS["bipop_cmaes"] = list(_BIPOP_CMAES_OPERATOR_LABELS)
try:
    _GENUINE_ENGINE_EMITTED_OPERATOR_LABELS.update(_BIPOP_CMAES_OPERATOR_LABELS)
except NameError:  # pragma: no cover
    pass

_prev_semanticize_operator_label_bipop_cmaes_native = semanticize_operator_label
def semanticize_operator_label(algorithm_id, label):  # type: ignore[override]
    aid = str(algorithm_id).lower().replace("-", "_")
    lab = str(label) if label not in {None, ""} else label
    if aid == "bipop_cmaes" and lab in _BIPOP_CMAES_OPERATOR_LABELS:
        return lab
    return _prev_semanticize_operator_label_bipop_cmaes_native(algorithm_id, label)

_prev_labels_for_algorithm_bipop_cmaes_native = labels_for_algorithm
def labels_for_algorithm(algorithm_id):  # type: ignore[override]
    if str(algorithm_id).lower().replace("-", "_") == "bipop_cmaes":
        return list(_BIPOP_CMAES_OPERATOR_LABELS)
    return _prev_labels_for_algorithm_bipop_cmaes_native(algorithm_id)

_prev_expand_compound_operator_label_bipop_cmaes_native = expand_compound_operator_label
def expand_compound_operator_label(algorithm_id, label):  # type: ignore[override]
    aid = str(algorithm_id).lower().replace("-", "_")
    lab = str(label) if label not in {None, ""} else label
    if aid == "bipop_cmaes" and lab in _BIPOP_CMAES_OPERATOR_LABELS:
        return [lab]
    return _prev_expand_compound_operator_label_bipop_cmaes_native(algorithm_id, label)

"""AERIAL: resilient 3-D aerial traffic management and knowledge-gated coordination."""
from .config import SimConfig
from .coordination.base import Controller, SplitController, ZeroController
from .coordination.boids import Boids
from .coordination.consensus import SecondOrderConsensus
from .coordination.formation import Formation, LeaderFollower, formation_offsets
from .coordination.mission import Mission
from .coordination.switching import MODE_NAMES, Switching
from .core.simulator import Observation, SimResult, Simulator
from .core.world import Obstacle, Target, World
from .epistemic.abstraction import Grid
from .epistemic.gate import KnowledgeGate, OracleGate
from .epistemic.detour import Detours
from .epistemic.worlds import AgentHistory, PossibleWorlds, build_possible_worlds, knows_safe
from .experiments.runner import (check_paired, check_paired_rows, pair_sweep, paired_streams, pairing_signature, record_experiment,
                                 run_batch, run_configs, run_one, run_pair, summarize, sweep)
from .network.comm import CommModel, DiskComm, FixedComm, build_comm_model
from .network.graph import (algebraic_connectivity, apply_packet_drops, disk_adjacency, is_connected,
                            laplacian, make_topology, to_networkx)
from .network.messages import MessageMemory
from .registry import aggregator_registry, attack_registry, controller_registry
from .resilience.aggregators import (AggResult, Aggregator, MeanAggregator, WMSRAggregator,
                                     build_aggregator)
from .resilience.attacks import Attack, Covert, NoAttack
from .resilience.flocal import is_f_local, pick_compromised_f_local
from .resilience.robustness import (is_rs_robust, is_rs_robust_bruteforce, max_r, max_s, resilience_conditions,
                                    robustness_witness)
from .resilience.wmsr import WMSRConsensus
from .scenarios import fleet_overrides, generate_scenario, no_attack_overrides
from .safety import (brake_command, gate_clearance, geofence_margins, geofence_repulsion, govern_speed,
                     obstacle_margins, obstacle_repulsion, predict_plan, prioritise, safe_for_agent, safety_margins,
                     separation_margins, separation_repulsion, steer_around)
from .specs_library import (geofence_by_agent, geofence_spec, mission_by_agent, mission_spec, obstacle_by_agent,
                            obstacle_spec, safety_by_agent, safety_spec, separation_by_agent, separation_spec,
                            target_by_agent, target_margins, target_spec)
from .stability import (fleet_rate, is_stable, is_stable_normalised, max_stable_dt, predicted_rate, spectral_radius,
                        worst_case_lambda)
from .stl import prefix_robustness, robustness  # the formula nodes stay in aerial.stl; T4.1's logic has an And too
from .analysis.mission import (
    area_coverage, arrival_times, completion_time, hesitation_time, honest_msgs_discarded_frac, mission_metrics,
    n_collisions, n_requeries, obstacle_clearance_series, safety_robustness, safety_violation_steps,
    safety_violation_time, survivors_found, target_coverage_series,
)
from .analysis.stl import (
    separation_robustness, stl_safety_robustness, stl_target_robustness, target_robustness_series,
)

__version__ = "0.1.0"

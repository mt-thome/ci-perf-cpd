"""Módulo público de experimentos e runners."""

from cpd.experiments.runner import (
    compute_effective_snr,
    compute_hazard_levels,
    derive_stable_seed,
    execute_single_cell,
    generate_scenario_series,
    get_git_commit_hash,
    instantiate_detector,
    save_partition_atomic,
)

__all__ = [
    "derive_stable_seed",
    "compute_effective_snr",
    "compute_hazard_levels",
    "generate_scenario_series",
    "instantiate_detector",
    "execute_single_cell",
    "save_partition_atomic",
    "get_git_commit_hash",
]

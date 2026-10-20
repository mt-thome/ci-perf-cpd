"""Submódulo de geradores de séries temporais sintéticas para benchmarks de CPD."""

from cpd.generators.synthetic import (
    NoiseType,
    SyntheticSeries,
    SyntheticSeriesConfig,
    generate_synthetic_series,
)

__all__ = [
    "NoiseType",
    "SyntheticSeries",
    "SyntheticSeriesConfig",
    "generate_synthetic_series",
]


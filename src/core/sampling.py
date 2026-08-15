"""Facade for backward compatibility with src.core.sampling."""

from ..sampling import (
    impute_missing_metrics,
    apply_skip_tags,
    sample_face_dataset,
    create_quality_tiers,
    filter_and_sample_by_quality,
)

__all__ = [
    "impute_missing_metrics",
    "apply_skip_tags",
    "sample_face_dataset",
    "create_quality_tiers",
    "filter_and_sample_by_quality",
]

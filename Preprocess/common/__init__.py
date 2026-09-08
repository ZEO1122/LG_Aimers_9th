"""Public APIs for the self-contained preprocessing package."""

from .base import CommandFeatureEngineer
from .domain import add_intent_features, add_semantic_features
from .history import (
    HISTORY_UNION_COLUMNS,
    PRIMARY_HISTORY_COLUMNS,
    SECONDARY_HISTORY_COLUMNS,
    HistoryFeatureBundle,
    build_history_feature_bundle,
    join_selected_pitcher_history,
)
from .mechanics import MechanicsBinner, build_mechanics_lookup
from .pipeline import PipelineConfig, PreprocessingPipeline
from .state import STATE_FEATURE_NAMES, build_state_features
from .tensor import DeepTensorEncoder

__all__ = [
    "CommandFeatureEngineer",
    "DeepTensorEncoder",
    "HISTORY_UNION_COLUMNS",
    "HistoryFeatureBundle",
    "MechanicsBinner",
    "PRIMARY_HISTORY_COLUMNS",
    "PipelineConfig",
    "PreprocessingPipeline",
    "SECONDARY_HISTORY_COLUMNS",
    "STATE_FEATURE_NAMES",
    "add_intent_features",
    "add_semantic_features",
    "build_history_feature_bundle",
    "build_mechanics_lookup",
    "build_state_features",
    "join_selected_pitcher_history",
]

"""Orchestrator for the parallel preprocessing branches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .base import CommandFeatureEngineer
from .domain import add_intent_features, add_semantic_features
from .history import (
    HISTORY_UNION_COLUMNS,
    PRIMARY_HISTORY_COLUMNS,
    HistoryFeatureBundle,
    build_history_feature_bundle,
    empty_history_lookup,
    join_selected_pitcher_history,
)
from .mechanics import MechanicsBinner, build_mechanics_lookup
from .state import build_state_features
from .tensor import DeepTensorEncoder

@dataclass(frozen=True)
class PipelineConfig:
    use_trackman: bool = True
    mapping_prefix_len: int = 50
    mapping_min_votes: int = 20
    mapping_min_purity: float = 0.99


class PreprocessingPipeline:
    """Fit train-only state once and transform every evaluation row independently."""

    def __init__(self, config: PipelineConfig = PipelineConfig()):
        self.config = config
        self.engineer = CommandFeatureEngineer()
        self.tensor_encoder = DeepTensorEncoder()
        self.mechanics_binner: MechanicsBinner | None = None
        self.history_table = empty_history_lookup(HISTORY_UNION_COLUMNS)
        self.history_bundle: HistoryFeatureBundle | None = None
        self.fitted_ = False

    def _join_history(self, rows: pd.DataFrame) -> pd.DataFrame:
        return join_selected_pitcher_history(
            rows, HISTORY_UNION_COLUMNS, self.history_table
        )

    def _deep_frame(
        self,
        rows: pd.DataFrame,
        base: pd.DataFrame,
        history_union: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[str]]:
        semantic = add_semantic_features(rows)
        history_primary = history_union[
            [*PRIMARY_HISTORY_COLUMNS, "tm_pitcher_history_available"]
        ]
        deep = pd.concat([base, history_primary, semantic], axis=1)
        categorical = [*self.base_categorical_, "sem_home_team"]
        return deep, categorical

    def fit(
        self,
        train_rows: pd.DataFrame,
        history_rows: pd.DataFrame | None = None,
    ) -> "PreprocessingPipeline":
        if "control_success" not in train_rows:
            raise ValueError("fit requires train rows with control_success")
        self.engineer.fit(train_rows)
        base, self.base_categorical_ = self.engineer.transform(train_rows)

        if self.config.use_trackman:
            if history_rows is None:
                raise ValueError("use_trackman=True requires trackman_history rows")
            min_history_year = int(pd.to_numeric(history_rows["season"]).min())
            max_train_year = int(pd.to_numeric(train_rows["season"]).max())
            cutoffs = tuple(range(min_history_year + 1, max_train_year + 2))
            self.history_bundle = build_history_feature_bundle(
                train_rows,
                history_rows,
                cutoff_years=cutoffs,
                prefix_len=self.config.mapping_prefix_len,
                min_votes=self.config.mapping_min_votes,
                min_purity=self.config.mapping_min_purity,
            )
            self.history_table = self.history_bundle.pitcher_features
            mechanics_lookup = build_mechanics_lookup(
                history_rows, self.history_bundle.entity_mapping, cutoffs
            )
            self.mechanics_binner = MechanicsBinner().fit(mechanics_lookup)

        history_union = self._join_history(train_rows)
        deep, categorical = self._deep_frame(train_rows, base, history_union)
        self.tensor_encoder.fit(deep, categorical)
        self.fitted_ = True
        return self

    def fit_transform(
        self,
        train_rows: pd.DataFrame,
        history_rows: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        self.fit(train_rows, history_rows)
        return self.transform(train_rows)

    def transform(self, rows: pd.DataFrame) -> dict[str, Any]:
        if not self.fitted_:
            raise RuntimeError("PreprocessingPipeline.fit must be called first")
        model_rows = rows.drop(columns=["control_success"], errors="ignore")
        base, _ = self.engineer.transform(model_rows)
        history_union = self._join_history(model_rows)
        semantic = add_semantic_features(model_rows)
        deep, _ = self._deep_frame(model_rows, base, history_union)
        intent = add_intent_features(model_rows)
        intent.index = model_rows.index
        result: dict[str, Any] = {
            "base": base,
            "history_union": history_union,
            "semantic": semantic,
            "deep_frame": deep,
            "intent_frame": intent,
            "state_features": build_state_features(base),
            "deep_tensor": self.tensor_encoder.transform(deep),
        }
        if self.mechanics_binner is not None:
            result["mechanics_bins"] = self.mechanics_binner.transform(model_rows)
        return result

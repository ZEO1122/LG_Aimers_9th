"""Train-fitted numeric standardization and categorical tokenization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence
import warnings

import numpy as np
import pandas as pd

def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


@dataclass
class DeepTensorEncoder:
    categorical_columns_: list[str] = field(default_factory=list)
    numeric_columns_: list[str] = field(default_factory=list)
    category_mappings_: dict[str, dict[Any, int]] = field(default_factory=dict)
    cardinalities_: list[int] = field(default_factory=list)
    numeric_mean_: np.ndarray | None = None
    numeric_std_: np.ndarray | None = None

    def fit(self, frame: pd.DataFrame, categorical_columns: Sequence[str]) -> "DeepTensorEncoder":
        self.categorical_columns_ = list(categorical_columns)
        self.numeric_columns_ = [c for c in frame.columns if c not in self.categorical_columns_]
        self.category_mappings_ = {}
        self.cardinalities_ = []
        for column in self.categorical_columns_:
            values = [_python_scalar(value) for value in frame[column].dropna().unique()]
            mapping = {value: token for token, value in enumerate(values, start=1)}
            self.category_mappings_[column] = mapping
            self.cardinalities_.append(len(mapping) + 1)

        raw = frame[self.numeric_columns_].to_numpy(np.float32)
        finite = np.where(np.isfinite(raw), raw, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean = np.nanmean(finite, axis=0).astype(np.float32)
            std = np.nanstd(finite, axis=0).astype(np.float32)
        self.numeric_mean_ = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
        self.numeric_std_ = np.where(
            np.isfinite(std) & (std >= 1e-6), std, 1.0
        ).astype(np.float32)
        return self

    def transform(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        if self.numeric_mean_ is None or self.numeric_std_ is None:
            raise RuntimeError("DeepTensorEncoder.fit을 먼저 실행해야 한다")
        numeric_raw = frame[self.numeric_columns_].to_numpy(np.float32)
        numeric = np.nan_to_num(
            (numeric_raw - self.numeric_mean_) / self.numeric_std_,
            nan=0.0,
            posinf=8.0,
            neginf=-8.0,
        )
        numeric = np.clip(numeric, -8.0, 8.0).astype(np.float16)
        categorical = np.zeros(
            (len(frame), len(self.categorical_columns_)), dtype=np.int32
        )
        for index, column in enumerate(self.categorical_columns_):
            categorical[:, index] = (
                frame[column].map(self.category_mappings_[column]).fillna(0).to_numpy(np.int32)
            )
        return {"numeric": numeric, "categorical": categorical}

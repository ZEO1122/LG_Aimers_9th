"""Label-free mechanics summaries and year-specific quartile bins."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

def build_mechanics_lookup(
    history: pd.DataFrame,
    entity_mapping: pd.DataFrame,
    cutoff_years: Sequence[int],
) -> pd.DataFrame:
    """세 mechanics 신호만 재생성한다. Residual correction은 만들지 않는다."""
    h = history.drop_duplicates(["season", "trackman_game_id", "pitch_no"], keep="first").copy()
    sign = h["pitcher_hand"].map({"Right": 1.0, "Left": -1.0}).astype(np.float64)
    h["arm_rel_side"] = pd.to_numeric(h["rel_side"], errors="coerce") * sign
    h["arm_horz_break"] = pd.to_numeric(h["horz_break"], errors="coerce") * sign
    for column in ("rel_height", "induced_vert_break", "rel_speed"):
        h[column] = pd.to_numeric(h[column], errors="coerce")

    accepted = entity_mapping.loc[
        entity_mapping["role"].eq("pitcher") & entity_mapping["accepted"].astype(bool)
    ].copy()
    accepted.rename(columns={
        "main_entity_id": "pitcher_id",
        "trackman_entity_id": "pitcher_trackman_id",
    }, inplace=True)

    outputs: list[pd.DataFrame] = []
    for cutoff in sorted(set(map(int, cutoff_years))):
        career = h.loc[h["season"].lt(cutoff)]
        recent = h.loc[h["season"].eq(cutoff - 1)]
        key = "pitcher_trackman_id"

        release = recent.groupby(key, sort=False).agg(
            recent1_rel_height_std=("rel_height", "std"),
            recent1_arm_rel_side_std=("arm_rel_side", "std"),
        )
        release["release_dispersion"] = np.sqrt(
            release["recent1_rel_height_std"].pow(2)
            + release["recent1_arm_rel_side_std"].pow(2)
        )

        movement_rows = []
        for pitcher_tm_id, group in recent.groupby(key, sort=False):
            pair = group[["induced_vert_break", "arm_horz_break"]].dropna()
            if len(pair) < 2:
                area = np.nan
            else:
                covariance = pair.cov(ddof=1).to_numpy(np.float64)
                area = float(np.sqrt(max(np.linalg.det(covariance), 0.0)))
            movement_rows.append((pitcher_tm_id, area))
        movement = pd.DataFrame(
            movement_rows, columns=[key, "movement_ellipse"]
        ).set_index(key) if movement_rows else pd.DataFrame(columns=["movement_ellipse"])

        arsenal = career.loc[career["pitch_type_group"].isin(["fastball", "breaking"])].groupby(
            [key, "pitch_type_group"], observed=True
        )["rel_speed"].mean().unstack()
        fast = arsenal["fastball"] if "fastball" in arsenal else pd.Series(index=arsenal.index, dtype=float)
        breaking = arsenal["breaking"] if "breaking" in arsenal else pd.Series(index=arsenal.index, dtype=float)
        gap = (fast - breaking).rename("fastball_breaking_velocity_gap")

        metrics = release[["release_dispersion"]].join(movement, how="outer").join(gap, how="outer")
        mapping = accepted.loc[accepted["cutoff_year"].eq(cutoff), ["pitcher_id", key]].copy()
        joined = mapping.set_index(key).join(metrics, how="left").reset_index(drop=True)
        joined.insert(0, "cutoff_year", cutoff)
        outputs.append(joined)

    result = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()
    if not result.empty and result.duplicated(["cutoff_year", "pitcher_id"]).any():
        raise ValueError("mechanics lookup key가 중복되어 있다")
    return result


@dataclass
class MechanicsBinner:
    thresholds_: dict[tuple[int, str], np.ndarray] = field(default_factory=dict)
    lookup_: pd.DataFrame | None = None

    def fit(self, lookup: pd.DataFrame) -> "MechanicsBinner":
        self.lookup_ = lookup.copy()
        for year in sorted(lookup["cutoff_year"].unique()):
            local = lookup.loc[lookup["cutoff_year"].eq(year)]
            for feature in (
                "release_dispersion",
                "movement_ellipse",
                "fastball_breaking_velocity_gap",
            ):
                values = pd.to_numeric(local[feature], errors="coerce").dropna().to_numpy(np.float64)
                self.thresholds_[(int(year), feature)] = (
                    np.quantile(values, [0.25, 0.50, 0.75]) if len(values) else np.array([])
                )
        return self

    def transform(self, rows: pd.DataFrame) -> pd.DataFrame:
        if self.lookup_ is None:
            raise RuntimeError("MechanicsBinner.fit을 먼저 실행해야 한다")
        keys = pd.DataFrame({
            "__row_order": np.arange(len(rows)),
            "cutoff_year": pd.to_numeric(rows["season"], errors="coerce").astype("Int64"),
            "pitcher_id": pd.to_numeric(rows["pitcher_id"], errors="coerce").astype("Int64"),
        })
        merged = keys.merge(
            self.lookup_, on=["cutoff_year", "pitcher_id"], how="left", sort=False, validate="many_to_one"
        ).sort_values("__row_order", kind="stable")
        output = pd.DataFrame(index=rows.index)
        for feature in ("release_dispersion", "movement_ellipse", "fastball_breaking_velocity_gap"):
            bins = np.full(len(rows), -1, dtype=np.int8)
            values = pd.to_numeric(merged[feature], errors="coerce").to_numpy(np.float64)
            years = pd.to_numeric(merged["cutoff_year"], errors="coerce").to_numpy(np.float64)
            for year in np.unique(years[np.isfinite(years)]).astype(int):
                mask = (years == year) & np.isfinite(values)
                thresholds = self.thresholds_.get((year, feature), np.array([]))
                if len(thresholds):
                    bins[mask] = np.searchsorted(thresholds, values[mask], side="right").astype(np.int8)
            output[f"{feature}_bin"] = bins
        return output

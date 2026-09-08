"""Common row-local feature engineering used by ML002 and later models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

TARGET_COLUMN = "control_success"
ID_COLUMN = "row_id"


@dataclass(frozen=True)
class _Rate:
    name: str
    column: str
    advances_with_target: bool = False


@dataclass(frozen=True)
class _Family:
    name: str
    prefix: str
    entity_column: str
    count_column: str
    rates: tuple[_Rate, ...]


_FAMILIES: tuple[_Family, ...] = (
    _Family(
        name="pitcher_outcome",
        prefix="p",
        entity_column="pitcher_id",
        count_column="asof_pitcher_n",
        rates=(
            _Rate("success", "asof_pitcher_success_rate", True),
            _Rate("reverse", "asof_pitcher_reverse_rate"),
            _Rate("middle", "asof_pitcher_middle_rate"),
            _Rate("ball", "asof_pitcher_ball_rate"),
            _Rate("strike", "asof_pitcher_strike_rate"),
        ),
    ),
    _Family(
        name="batter_outcome",
        prefix="b",
        entity_column="batter_id",
        count_column="asof_batter_n",
        rates=(
            _Rate("success", "asof_batter_success_rate", True),
            _Rate("middle", "asof_batter_middle_rate"),
        ),
    ),
    _Family(
        name="pitcher_pitchmix",
        prefix="pmix",
        entity_column="pitcher_id",
        count_column="asof_pitcher_pitchmix_n",
        rates=(
            _Rate("fastball", "asof_pitcher_fastball_rate"),
            _Rate("breaking", "asof_pitcher_breaking_rate"),
            _Rate("offspeed", "asof_pitcher_offspeed_rate"),
        ),
    ),
)


_RATE_DEFAULTS: Mapping[str, float] = {
    "asof_pitcher_success_rate": 0.50,
    "asof_pitcher_reverse_rate": 0.22,
    "asof_pitcher_middle_rate": 0.15,
    "asof_pitcher_ball_rate": 0.37,
    "asof_pitcher_strike_rate": 0.44,
    "asof_batter_success_rate": 0.50,
    "asof_batter_middle_rate": 0.15,
    "asof_pitcher_fastball_rate": 0.55,
    "asof_pitcher_breaking_rate": 0.29,
    "asof_pitcher_offspeed_rate": 0.16,
}


_RECENT_RATE_COLUMNS: tuple[str, ...] = (
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
)


_MISSING_FLAG_COLUMNS: tuple[str, ...] = (
    *(rate.column for family in _FAMILIES for rate in family.rates),
    *_RECENT_RATE_COLUMNS,
    "home_win_expectancy",
    "away_win_expectancy",
    "li",
)


_RAW_CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_id",
    "batter_id",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)


def _safe_logit(probability: np.ndarray, epsilon: float) -> np.ndarray:
    clipped = np.clip(probability, epsilon, 1.0 - epsilon)
    return np.log(clipped) - np.log1p(-clipped)


def _encode_fixed(series: pd.Series, mapping: Mapping[str, int]) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series.dtype):
        return pd.to_numeric(series, errors="coerce").fillna(-1).to_numpy(np.int32)
    normalized = series.astype("string").str.strip()
    return normalized.map(mapping).fillna(-1).to_numpy(np.int32)


def _encode_base_state(series: pd.Series) -> np.ndarray:
    return _encode_fixed(
        series,
        {
            "___": 0,
            "1__": 1,
            "_2_": 2,
            "12_": 3,
            "__3": 4,
            "1_3": 5,
            "_23": 6,
            "123": 7,
        },
    )


class CommandFeatureEngineer:
    """Leakage-safe fit/transform feature builder.

    ``transform`` returns ``(frame, categorical_columns)``.  Every column in the
    returned frame is numeric and stored as ``float32`` or ``int32``.  Columns
    named in ``categorical_columns`` contain deterministic integer tokens and
    can be passed as native categorical features to CatBoost/LightGBM.

    """

    def __init__(
        self,
        *,
        career_prior_strength: float = 100.0,
        pitcher_ytd_prior_strength: float = 50.0,
        batter_ytd_prior_strength: float = 100.0,
        pitchmix_ytd_prior_strength: float = 50.0,
        epsilon: float = 1e-4,
    ) -> None:
        for name, value in {
            "career_prior_strength": career_prior_strength,
            "pitcher_ytd_prior_strength": pitcher_ytd_prior_strength,
            "batter_ytd_prior_strength": batter_ytd_prior_strength,
            "pitchmix_ytd_prior_strength": pitchmix_ytd_prior_strength,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 < epsilon < 0.5:
            raise ValueError("epsilon must lie in (0, 0.5)")

        self.career_prior_strength = float(career_prior_strength)
        self.pitcher_ytd_prior_strength = float(pitcher_ytd_prior_strength)
        self.batter_ytd_prior_strength = float(batter_ytd_prior_strength)
        self.pitchmix_ytd_prior_strength = float(pitchmix_ytd_prior_strength)
        self.epsilon = float(epsilon)

    @property
    def is_fitted(self) -> bool:
        return hasattr(self, "snapshots_")

    def fit(
        self,
        train: pd.DataFrame,
        y: Sequence[float] | pd.Series | np.ndarray | None = None,
    ) -> "CommandFeatureEngineer":
        """Fit historical season-start states from labelled rows.

        Supplying all 2019--2024 rows is safe for ``fit_transform``: when a 2024
        row is transformed, the lookup explicitly ignores snapshots from 2024
        and later.  A conventional forward fold may instead call ``fit`` on
        rows before the validation year; both routes produce the same 2024
        snapshot features.
        """

        required = {"season"}
        for family in _FAMILIES:
            required.update((family.entity_column, family.count_column))
            required.update(rate.column for rate in family.rates)
        _require_columns(train, required, "train")

        if y is None:
            if TARGET_COLUMN not in train.columns:
                raise ValueError(
                    f"fit requires y or a {TARGET_COLUMN!r} column to close "
                    "season-end success snapshots"
                )
            target = pd.to_numeric(train[TARGET_COLUMN], errors="raise").to_numpy(
                dtype=np.float64
            )
        else:
            target = np.asarray(y, dtype=np.float64)
            if target.shape != (len(train),):
                raise ValueError(
                    f"y must have shape ({len(train)},), got {target.shape}"
                )
        if not np.isfinite(target).all() or not np.isin(target, (0.0, 1.0)).all():
            raise ValueError("target must contain only finite binary values")

        self.input_feature_columns_ = [
            column
            for column in train.columns
            if column not in (ID_COLUMN, TARGET_COLUMN)
        ]
        self.fit_min_season_ = int(pd.to_numeric(train["season"]).min())
        self.fit_max_season_ = int(pd.to_numeric(train["season"]).max())

        self.snapshots_: dict[str, pd.DataFrame] = {}
        for family in _FAMILIES:
            self.snapshots_[family.name] = self._build_snapshots(
                train, target, family
            )

        seasons = pd.to_numeric(train["season"], errors="raise").to_numpy(np.int32)
        self.target_prior_stats_ = (
            pd.DataFrame({"season": seasons, "value": target})
            .groupby("season", sort=True)["value"]
            .agg(["sum", "count"])
            .reset_index()
        )
        self.rate_prior_stats_: dict[str, pd.DataFrame] = {}
        for column in dict.fromkeys(
            rate.column for family in _FAMILIES for rate in family.rates
        ):
            value = _numeric(train, column)
            valid = np.isfinite(value)
            stats = pd.DataFrame(
                {"season": seasons[valid], "value": value[valid]}
            )
            self.rate_prior_stats_[column] = (
                stats.groupby("season", sort=True)["value"]
                .agg(["sum", "count"])
                .reset_index()
            )

        self.categorical_columns_ = [
            column
            for column in _RAW_CATEGORICAL_COLUMNS
            if column in self.input_feature_columns_
        ] + [
            "game_month_cat",
            "game_dayofweek_cat",
            "inning_bucket_cat",
            "count_state_cat",
            "hand_matchup_cat",
            "team_matchup_cat",
            "score_state_cat",
        ]
        return self

    def fit_transform(
        self,
        train: pd.DataFrame,
        y: Sequence[float] | pd.Series | np.ndarray | None = None,
    ) -> tuple[pd.DataFrame, list[str]]:
        return self.fit(train, y=y).transform(train)

    def transform(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Transform rows independently using fitted historical state only."""

        if not self.is_fitted:
            raise RuntimeError("CommandFeatureEngineer must be fitted before transform")
        _require_columns(frame, self.input_feature_columns_, "frame")

        raw: dict[str, np.ndarray] = {}
        for column in self.input_feature_columns_:
            series = frame[column]
            if column == "top_bottom":
                raw[column] = _encode_fixed(
                    series, {"T": 1, "Top": 1, "B": 0, "Bottom": 0}
                )
            elif column == "game_type":
                raw[column] = _encode_fixed(series, {"R": 0, "F": 1})
            elif column == "base_state":
                raw[column] = _encode_base_state(series)
            elif column in _RAW_CATEGORICAL_COLUMNS:
                raw[column] = (
                    pd.to_numeric(series, errors="coerce")
                    .fillna(-1)
                    .to_numpy(np.int32)
                )
            elif pd.api.types.is_numeric_dtype(series.dtype):
                values = pd.to_numeric(series, errors="coerce")
                if pd.api.types.is_integer_dtype(values.dtype) and not values.isna().any():
                    raw[column] = values.to_numpy(np.int32)
                else:
                    raw[column] = values.to_numpy(np.float32)
            else:
                raise TypeError(
                    f"unexpected non-numeric input column {column!r}; add a "
                    "deterministic encoder before using it"
                )

        engineered: dict[str, np.ndarray] = {}
        seasons = pd.to_numeric(frame["season"], errors="raise").to_numpy(np.int32)
        for family in _FAMILIES:
            self._engineer_family(frame, seasons, family, engineered)

        self._engineer_recent_form(frame, engineered)
        self._engineer_context(frame, raw, engineered)
        self._engineer_pitchmix(engineered)

        for column in _MISSING_FLAG_COLUMNS:
            if column in frame.columns:
                engineered[f"{column}_missing"] = (~np.isfinite(_numeric(frame, column))).astype(
                    np.int32
                )

        output = pd.DataFrame({**raw, **engineered}, index=frame.index)
        categorical = [c for c in self.categorical_columns_ if c in output.columns]
        categorical_set = set(categorical)
        for column in output.columns:
            if column in categorical_set:
                output[column] = (
                    pd.to_numeric(output[column], errors="coerce")
                    .fillna(-1)
                    .to_numpy(np.int32)
                )
            elif pd.api.types.is_integer_dtype(output[column].dtype):
                output[column] = output[column].to_numpy(np.int32)
            else:
                output[column] = pd.to_numeric(
                    output[column], errors="coerce"
                ).to_numpy(np.float32)
        return output, categorical

    def _build_snapshots(
        self,
        train: pd.DataFrame,
        target: np.ndarray,
        family: _Family,
    ) -> pd.DataFrame:
        columns = [
            "season",
            family.entity_column,
            family.count_column,
            *(rate.column for rate in family.rates),
        ]
        work = train.loc[:, columns].copy().reset_index(drop=True)
        work["__target"] = target
        work["season"] = pd.to_numeric(work["season"], errors="raise").astype(np.int32)
        work[family.entity_column] = pd.to_numeric(
            work[family.entity_column], errors="raise"
        ).astype(np.int64)
        work[family.count_column] = pd.to_numeric(
            work[family.count_column], errors="raise"
        )

        last_positions = work.groupby(
            [family.entity_column, "season"], sort=False, observed=True
        )[family.count_column].idxmax()
        last = work.loc[last_positions].copy()
        snapshot = last[[family.entity_column, "season"]].rename(
            columns={"season": "source_season"}
        )

        n = pd.to_numeric(last[family.count_column], errors="coerce").to_numpy(
            np.float64
        )
        for rate in family.rates:
            probability = pd.to_numeric(last[rate.column], errors="coerce").to_numpy(
                np.float64
            )
            valid = np.isfinite(n) & (n >= 0) & (np.isfinite(probability) | (n == 0))
            count = np.rint(np.nan_to_num(probability, nan=0.0) * np.maximum(n, 0.0))
            base_n = np.maximum(n, 0.0)
            base_count = count
            if rate.advances_with_target:
                base_n = base_n + 1.0
                base_count = base_count + last["__target"].to_numpy(np.float64)
            snapshot[f"{rate.name}__base_n"] = np.rint(base_n).astype(np.int32)
            snapshot[f"{rate.name}__base_count"] = np.rint(base_count).astype(
                np.int32
            )
            snapshot[f"{rate.name}__valid"] = valid.astype(np.int32)

        return snapshot.sort_values(
            [family.entity_column, "source_season"], kind="stable"
        ).reset_index(drop=True)

    def _lookup_snapshots(
        self,
        family: _Family,
        seasons: np.ndarray,
        entity_values: np.ndarray,
    ) -> dict[str, np.ndarray]:
        snapshot = self.snapshots_[family.name]
        result: dict[str, np.ndarray] = {
            "source_season": np.full(len(seasons), -1, dtype=np.int32)
        }
        value_columns = [
            column
            for column in snapshot.columns
            if column not in (family.entity_column, "source_season")
        ]
        for column in value_columns:
            result[column] = np.zeros(len(seasons), dtype=np.int32)

        # The loop only chooses among fitted tables.  No statistic is calculated
        # from rows sharing the same transform batch.
        for season in np.unique(seasons):
            positions = np.flatnonzero(seasons == season)
            eligible = snapshot.loc[snapshot["source_season"] < season]
            if eligible.empty:
                continue
            latest = eligible.drop_duplicates(family.entity_column, keep="last").set_index(
                family.entity_column
            )
            matched = latest.reindex(entity_values[positions])
            source = matched["source_season"].fillna(-1).to_numpy(np.int32)
            result["source_season"][positions] = source
            for column in value_columns:
                result[column][positions] = matched[column].fillna(0).to_numpy(np.int32)
        return result

    def _rate_prior(self, column: str, seasons: np.ndarray) -> np.ndarray:
        stats = self.rate_prior_stats_[column]
        result = np.empty(len(seasons), dtype=np.float64)
        fallback = float(_RATE_DEFAULTS[column])
        for season in np.unique(seasons):
            previous = stats.loc[stats["season"] < season]
            if previous.empty:
                value = fallback
            else:
                count = float(previous["count"].sum())
                value = float(previous["sum"].sum() / count) if count else fallback
            result[seasons == season] = value
        return result

    def _family_ytd_strength(self, family: _Family) -> float:
        if family.prefix == "p":
            return self.pitcher_ytd_prior_strength
        if family.prefix == "b":
            return self.batter_ytd_prior_strength
        return self.pitchmix_ytd_prior_strength

    def _engineer_family(
        self,
        frame: pd.DataFrame,
        seasons: np.ndarray,
        family: _Family,
        output: dict[str, np.ndarray],
    ) -> None:
        entity = (
            pd.to_numeric(frame[family.entity_column], errors="coerce")
            .fillna(-1)
            .to_numpy(np.int64)
        )
        n_float = _numeric(frame, family.count_column)
        n = np.rint(np.maximum(np.nan_to_num(n_float, nan=0.0), 0.0)).astype(np.int32)
        lookup = self._lookup_snapshots(family, seasons, entity)
        source = lookup["source_season"]
        has_snapshot = source >= 0
        career_alpha = self.career_prior_strength
        ytd_alpha = self._family_ytd_strength(family)

        output[f"{family.prefix}_career_n"] = n
        output[f"{family.prefix}_career_log_n"] = np.log1p(n).astype(np.float32)
        output[f"{family.prefix}_career_reliability"] = (
            n / (n + career_alpha)
        ).astype(np.float32)
        output[f"{family.prefix}_snapshot_available"] = has_snapshot.astype(np.int32)
        output[f"{family.prefix}_snapshot_season"] = source
        output[f"{family.prefix}_snapshot_age"] = np.where(
            has_snapshot, seasons - source, -1
        ).astype(np.int32)

        first_ytd_n: np.ndarray | None = None
        for rate in family.rates:
            probability = _numeric(frame, rate.column)
            prior = self._rate_prior(rate.column, seasons)
            rate_valid = np.isfinite(probability) | (n == 0)
            current_count = np.rint(
                np.nan_to_num(probability, nan=0.0) * n.astype(np.float64)
            )
            current_count = np.clip(current_count, 0.0, n.astype(np.float64))

            base_valid = lookup[f"{rate.name}__valid"].astype(bool) & has_snapshot
            base_n = np.where(
                base_valid, lookup[f"{rate.name}__base_n"], 0
            ).astype(np.int32)
            base_count = np.where(
                base_valid, lookup[f"{rate.name}__base_count"], 0
            ).astype(np.int32)

            career_smooth = (
                current_count + career_alpha * prior
            ) / (n + career_alpha)
            career_smooth = np.clip(career_smooth, self.epsilon, 1.0 - self.epsilon)
            start_smooth = (
                base_count.astype(np.float64) + career_alpha * prior
            ) / (base_n + career_alpha)

            delta_n = n.astype(np.int64) - base_n.astype(np.int64)
            delta_count = current_count - base_count.astype(np.float64)
            invalid = (
                (~rate_valid)
                | (delta_n < 0)
                | (delta_count < -0.25)
                | (delta_count > delta_n + 0.25)
            )
            safe_n = np.where(invalid, 0, delta_n).astype(np.int32)
            safe_count = np.where(
                invalid, 0.0, np.clip(delta_count, 0.0, np.maximum(delta_n, 0))
            )
            raw_ytd = np.divide(
                safe_count,
                safe_n,
                out=np.full(len(frame), np.nan, dtype=np.float64),
                where=safe_n > 0,
            )
            ytd_smooth = (
                safe_count + ytd_alpha * start_smooth
            ) / (safe_n + ytd_alpha)
            ytd_smooth = np.clip(ytd_smooth, self.epsilon, 1.0 - self.epsilon)

            stem = f"{family.prefix}_{rate.name}"
            output[f"{stem}_career_count"] = current_count.astype(np.int32)
            output[f"{stem}_career_rate_smooth"] = career_smooth.astype(np.float32)
            output[f"{stem}_career_logit"] = _safe_logit(
                career_smooth, self.epsilon
            ).astype(np.float32)
            output[f"{stem}_ytd_n"] = safe_n
            output[f"{stem}_ytd_count"] = safe_count.astype(np.int32)
            output[f"{stem}_ytd_rate_raw"] = raw_ytd.astype(np.float32)
            output[f"{stem}_ytd_rate_smooth"] = ytd_smooth.astype(np.float32)
            output[f"{stem}_ytd_reliability"] = (
                safe_n / (safe_n + ytd_alpha)
            ).astype(np.float32)
            output[f"{stem}_ytd_delta_career"] = (
                ytd_smooth - career_smooth
            ).astype(np.float32)
            output[f"{stem}_ytd_invalid"] = invalid.astype(np.int32)
            if first_ytd_n is None:
                first_ytd_n = safe_n

        if first_ytd_n is not None:
            output[f"{family.prefix}_ytd_n"] = first_ytd_n
            output[f"{family.prefix}_ytd_log_n"] = np.log1p(first_ytd_n).astype(
                np.float32
            )

    def _engineer_recent_form(
        self, frame: pd.DataFrame, output: dict[str, np.ndarray]
    ) -> None:
        career_success = output["p_success_career_rate_smooth"].astype(np.float64)
        career_middle = output["p_middle_career_rate_smooth"].astype(np.float64)
        for window in (1, 3, 5):
            success = _numeric(frame, f"asof_pitcher_prev{window}_game_success_rate")
            middle = _numeric(frame, f"asof_pitcher_prev{window}_game_middle_rate")
            output[f"p_prev{window}_success_delta_career"] = (
                success - career_success
            ).astype(np.float32)
            output[f"p_prev{window}_middle_delta_career"] = (
                middle - career_middle
            ).astype(np.float32)
        output["p_recent_success_slope_1v5"] = (
            _numeric(frame, "asof_pitcher_prev1_game_success_rate")
            - _numeric(frame, "asof_pitcher_prev5_game_success_rate")
        ).astype(np.float32)
        output["p_recent_middle_slope_1v5"] = (
            _numeric(frame, "asof_pitcher_prev1_game_middle_rate")
            - _numeric(frame, "asof_pitcher_prev5_game_middle_rate")
        ).astype(np.float32)

    def _engineer_context(
        self,
        frame: pd.DataFrame,
        raw: dict[str, np.ndarray],
        output: dict[str, np.ndarray],
    ) -> None:
        balls = _numeric(frame, "balls_before")
        strikes = _numeric(frame, "strikes_before")
        outs = _numeric(frame, "outs_before")
        inning = _numeric(frame, "inning")
        month = _numeric(frame, "game_month")
        runners = _numeric(frame, "num_runners_on")
        r1 = _numeric(frame, "runner_on_1b")
        r2 = _numeric(frame, "runner_on_2b")
        r3 = _numeric(frame, "runner_on_3b")
        li = np.nan_to_num(_numeric(frame, "li"), nan=0.0)
        score = np.nan_to_num(_numeric(frame, "score_diff_pitcher_team"), nan=0.0)
        top = raw["top_bottom"].astype(np.int32)
        pitcher_hand = raw["pitcher_hand"].astype(np.int32)
        batter_hand = raw["batter_hand"].astype(np.int32)
        pitcher_team = raw["pitcher_team_id"].astype(np.int32)
        batter_team = raw["batter_team_id"].astype(np.int32)

        output["game_month_cat"] = np.nan_to_num(month, nan=-1).astype(np.int32)
        output["game_dayofweek_cat"] = np.nan_to_num(
            _numeric(frame, "game_dayofweek"), nan=-1
        ).astype(np.int32)
        output["inning_bucket_cat"] = np.where(
            inning >= 10, 10, np.nan_to_num(inning, nan=-1)
        ).astype(np.int32)
        output["count_state_cat"] = np.where(
            np.isfinite(balls) & np.isfinite(strikes), balls * 3 + strikes, -1
        ).astype(np.int32)
        output["count_ball_minus_strike"] = (balls - strikes).astype(np.float32)
        output["count_pitcher_ahead"] = (strikes > balls).astype(np.int32)
        output["count_hitter_ahead"] = (balls > strikes).astype(np.int32)
        output["count_full"] = ((balls == 3) & (strikes == 2)).astype(np.int32)
        output["count_three_ball"] = (balls == 3).astype(np.int32)
        output["count_two_strike"] = (strikes == 2).astype(np.int32)
        output["count_first_pitch"] = ((balls == 0) & (strikes == 0)).astype(np.int32)
        output["outs_plus_runners"] = np.nan_to_num(outs + runners, nan=0).astype(
            np.float32
        )

        output["base_bitmask"] = np.nan_to_num(r1 + 2 * r2 + 4 * r3, nan=-1).astype(
            np.int32
        )
        output["runner_in_scoring_position"] = ((r2 > 0) | (r3 > 0)).astype(np.int32)
        output["bases_loaded"] = ((r1 > 0) & (r2 > 0) & (r3 > 0)).astype(np.int32)
        output["runner_pressure"] = (np.nan_to_num(runners, nan=0) * li).astype(
            np.float32
        )

        output["hand_matchup_cat"] = (pitcher_hand * 10 + batter_hand).astype(np.int32)
        output["same_hand_matchup"] = (pitcher_hand == batter_hand).astype(np.int32)
        output["team_matchup_cat"] = (
            pitcher_team.astype(np.int64) * 100 + batter_team.astype(np.int64)
        ).astype(np.int32)

        output["score_state_cat"] = np.sign(score).astype(np.int32)
        output["score_abs_diff"] = np.abs(score).astype(np.float32)
        output["score_tied"] = (score == 0).astype(np.int32)
        output["score_close_1"] = (np.abs(score) <= 1).astype(np.int32)
        output["score_close_2"] = (np.abs(score) <= 2).astype(np.int32)
        late = inning >= 7
        output["late_inning"] = late.astype(np.int32)
        output["extra_innings"] = (inning >= 10).astype(np.int32)
        output["late_close_game"] = (late & (np.abs(score) <= 2)).astype(np.int32)

        home_we = _numeric(frame, "home_win_expectancy")
        away_we = _numeric(frame, "away_win_expectancy")
        pitcher_we = np.where(top == 1, home_we, away_we)
        output["pitcher_team_win_expectancy"] = pitcher_we.astype(np.float32)
        output["pitcher_team_win_edge"] = ((pitcher_we - 50.0) / 50.0).astype(
            np.float32
        )
        output["win_expectancy_closeness"] = np.clip(
            1.0 - np.abs(pitcher_we - 50.0) / 50.0, 0.0, 1.0
        ).astype(np.float32)

        output["li_log1p"] = np.log1p(np.maximum(li, 0.0)).astype(np.float32)
        output["high_leverage"] = (li >= 1.5).astype(np.int32)
        output["li_x_late"] = (li * late).astype(np.float32)
        output["li_x_close"] = (li * (np.abs(score) <= 2)).astype(np.float32)
        output["pressure_index"] = (
            li * (1.0 + np.nan_to_num(runners, nan=0.0)) * (1.0 + late.astype(float))
        ).astype(np.float32)

        angle = 2.0 * np.pi * (month - 1.0) / 12.0
        output["month_sin"] = np.sin(angle).astype(np.float32)
        output["month_cos"] = np.cos(angle).astype(np.float32)
        season = _numeric(frame, "season")
        output["season_since_2019"] = (season - 2019.0).astype(np.float32)
        output["post_2022_regime"] = (season >= 2023).astype(np.int32)
        game_type = raw["game_type"].astype(np.int32)
        output["post_2022_x_game_type_f"] = (
            (season >= 2023) & (game_type == 1)
        ).astype(np.int32)

    def _engineer_pitchmix(self, output: dict[str, np.ndarray]) -> None:
        career = np.column_stack(
            [
                output[f"pmix_{name}_career_rate_smooth"]
                for name in ("fastball", "breaking", "offspeed")
            ]
        ).astype(np.float64)
        ytd = np.column_stack(
            [
                output[f"pmix_{name}_ytd_rate_smooth"]
                for name in ("fastball", "breaking", "offspeed")
            ]
        ).astype(np.float64)
        for label, matrix in (("career", career), ("ytd", ytd)):
            total = np.maximum(matrix.sum(axis=1, keepdims=True), self.epsilon)
            normalized = np.clip(matrix / total, self.epsilon, 1.0)
            output[f"pmix_{label}_entropy"] = (
                -np.sum(normalized * np.log(normalized), axis=1)
            ).astype(np.float32)
            output[f"pmix_{label}_concentration"] = np.max(
                normalized, axis=1
            ).astype(np.float32)
            output[f"pmix_{label}_rate_sum"] = matrix.sum(axis=1).astype(np.float32)


__all__ = ["CommandFeatureEngineer", "TARGET_COLUMN", "ID_COLUMN"]

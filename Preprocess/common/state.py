"""Pure DataFrame implementation of the 23 row-local state features."""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_STATE_COLUMNS: tuple[str, ...] = (
    "balls_before",
    "strikes_before",
    "p_success_ytd_n",
    "p_success_ytd_count",
    "p_success_ytd_rate_smooth",
    "b_success_ytd_n",
    "b_success_ytd_count",
    "b_success_ytd_rate_smooth",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_pitcher_prev1_game_success_rate_missing",
    "asof_pitcher_prev3_game_success_rate_missing",
    "asof_pitcher_prev5_game_success_rate_missing",
    "asof_pitcher_prev1_game_middle_rate_missing",
    "asof_pitcher_prev3_game_middle_rate_missing",
    "asof_pitcher_prev5_game_middle_rate_missing",
    "p_middle_career_rate_smooth",
    "pmix_fastball_ytd_delta_career",
    "pmix_breaking_ytd_delta_career",
    "pmix_offspeed_ytd_delta_career",
    "pmix_fastball_ytd_reliability",
)

STATE_FEATURE_NAMES: tuple[str, ...] = (
    "pitcher_preseason_minus_half",
    "pitcher_ytd_state_delta",
    "batter_preseason_minus_half",
    "batter_ytd_state_delta",
    "recent_success_delta",
    "recent_middle_delta",
    "reliable_pitchmix_drift",
    "pitcher_ytd_reliability",
    "batter_ytd_reliability",
    "pitcher_state_x_pitchmix_drift",
    "recent_delta_reliability_gated",
    *(f"pitcher_state_x_count_{code}" for code in range(12)),
)


def _column(frame: pd.DataFrame, name: str) -> np.ndarray:
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(np.float64)


def build_state_features(engineered: pd.DataFrame) -> pd.DataFrame:
    """Create the 23 state features without model artifacts or neighbouring rows."""

    missing = sorted(set(REQUIRED_STATE_COLUMNS).difference(engineered.columns))
    if missing:
        raise KeyError(f"missing state-feature columns: {missing}")

    raw = lambda name: _column(engineered, name)
    p_n = np.maximum(np.rint(raw("p_success_ytd_n")), 0.0)
    p_s = np.clip(np.rint(raw("p_success_ytd_count")), 0.0, p_n)
    b_n = np.maximum(np.rint(raw("b_success_ytd_n")), 0.0)
    b_s = np.clip(np.rint(raw("b_success_ytd_count")), 0.0, b_n)
    p_smooth = raw("p_success_ytd_rate_smooth")
    b_smooth = raw("b_success_ytd_rate_smooth")
    p_start = np.clip(((p_n + 50.0) * p_smooth - p_s) / 50.0, 0.20, 0.80)
    b_start = np.clip(((b_n + 100.0) * b_smooth - b_s) / 100.0, 0.20, 0.80)
    theta_p = (p_s + 50.0 * p_start) / (p_n + 50.0)
    theta_b = (b_s + 100.0 * b_start) / (b_n + 100.0)
    p_rel = p_n / (p_n + 50.0)
    b_rel = b_n / (b_n + 100.0)

    recent_success: list[np.ndarray] = []
    recent_middle: list[np.ndarray] = []
    career_middle = raw("p_middle_career_rate_smooth")
    for window in (1, 3, 5):
        success = raw(f"asof_pitcher_prev{window}_game_success_rate")
        middle = raw(f"asof_pitcher_prev{window}_game_middle_rate")
        success_missing = raw(
            f"asof_pitcher_prev{window}_game_success_rate_missing"
        ) > 0.5
        middle_missing = raw(
            f"asof_pitcher_prev{window}_game_middle_rate_missing"
        ) > 0.5
        recent_success.append(np.where(success_missing, p_start, success))
        recent_middle.append(np.where(middle_missing, career_middle, middle))

    recent = (
        0.50 * recent_success[0]
        + 0.30 * recent_success[1]
        + 0.20 * recent_success[2]
    )
    recent_mid = (
        0.50 * recent_middle[0]
        + 0.30 * recent_middle[1]
        + 0.20 * recent_middle[2]
    )
    recent_delta = recent - theta_p
    middle_delta = recent_mid - career_middle
    pmix_drift = np.clip(
        np.abs(raw("pmix_fastball_ytd_delta_career"))
        + np.abs(raw("pmix_breaking_ytd_delta_career"))
        + np.abs(raw("pmix_offspeed_ytd_delta_career")),
        0.0,
        1.5,
    )
    pmix_rel = np.clip(raw("pmix_fastball_ytd_reliability"), 0.0, 1.0)
    reliable_drift = pmix_rel * pmix_drift
    state_delta = theta_p - p_start
    count_state = (
        np.rint(raw("balls_before")).astype(np.int32) * 3
        + np.rint(raw("strikes_before")).astype(np.int32)
    )

    values: list[np.ndarray] = [
        p_start - 0.5,
        state_delta,
        b_start - 0.5,
        theta_b - b_start,
        recent_delta,
        middle_delta,
        reliable_drift,
        p_rel,
        b_rel,
        state_delta * reliable_drift,
        recent_delta * (1.0 - 0.50 * p_rel),
    ]
    values.extend(state_delta * (count_state == code) for code in range(12))
    matrix = np.column_stack(values).astype(np.float64)
    if matrix.shape[1] != len(STATE_FEATURE_NAMES):
        raise AssertionError("state feature schema width mismatch")
    if not np.isfinite(matrix).all():
        raise ValueError("state features contain non-finite values")
    return pd.DataFrame(matrix, columns=STATE_FEATURE_NAMES, index=engineered.index)

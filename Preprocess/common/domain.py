"""Row-local baseball semantic and intent feature branches."""

from __future__ import annotations

from typing import Iterable
import warnings

import numpy as np
import pandas as pd

def add_semantic_features(rows: pd.DataFrame) -> pd.DataFrame:
    """Baseball state features copied exactly from the frozen 2024 builder."""

    balls = rows["balls_before"].to_numpy(np.int8)
    strikes = rows["strikes_before"].to_numpy(np.int8)
    outs = rows["outs_before"].to_numpy(np.int8)
    inning = rows["inning"].to_numpy(np.int16)
    r1 = rows["runner_on_1b"].to_numpy(np.int8)
    r2 = rows["runner_on_2b"].to_numpy(np.int8)
    r3 = rows["runner_on_3b"].to_numpy(np.int8)
    score = rows["score_diff_pitcher_team"].to_numpy(np.float32)
    leverage = rows["li"].fillna(0).to_numpy(np.float32)

    first_open = r1 == 0
    force_length = r1 + (r1 & r2) + (r1 & r2 & r3)
    walk_run = (r1 & r2 & r3).astype(np.float32)
    double_play = ((r1 == 1) & (outs < 2)).astype(np.float32)
    steal_pressure = ((r1 == 1) & (r2 == 0) & (outs < 2)).astype(np.float32)
    bunt_state = ((outs == 0) & ((r1 == 1) | (r2 == 1)) & (np.abs(score) <= 2))
    sac_fly = ((r3 == 1) & (outs < 2)).astype(np.float32)
    pitch_around = (
        first_open & ((r2 == 1) | (r3 == 1)) & (inning >= 7) & (np.abs(score) <= 2)
    )
    must_strike = balls == 3
    finish = strikes == 2
    attack_pressure = (
        1.5 * must_strike
        + walk_run
        + 0.35 * force_length
        + leverage * (balls >= 2)
    )
    expand_freedom = (
        (strikes > balls).astype(np.float32)
        + 0.5 * (first_open & (r2 == 0) & (r3 == 0))
        - 0.5 * leverage
    )
    home_team = np.where(
        rows["top_bottom"].eq("T").to_numpy(),
        rows["pitcher_team_id"].to_numpy(),
        rows["batter_team_id"].to_numpy(),
    )

    return pd.DataFrame(
        {
            "sem_first_base_open": first_open.astype(np.float32),
            "sem_force_length": force_length.astype(np.float32),
            "sem_walk_run": walk_run,
            "sem_double_play": double_play,
            "sem_steal_pressure": steal_pressure,
            "sem_bunt_state": bunt_state.astype(np.float32),
            "sem_sac_fly": sac_fly,
            "sem_pitch_around": pitch_around.astype(np.float32),
            "sem_must_strike": must_strike.astype(np.float32),
            "sem_finish": finish.astype(np.float32),
            "sem_attack_pressure": attack_pressure.astype(np.float32),
            "sem_expand_freedom": expand_freedom.astype(np.float32),
            "sem_home_team": home_team.astype(np.float32),
            "sem_futures_post_break": (
                rows["game_type"].eq("F").to_numpy()
                & rows["season"].ge(2023).to_numpy()
            ).astype(np.float32),
        },
        index=rows.index,
    )

RAW_NUMERIC = (
    "season",
    "game_month",
    "game_dayofweek",
    "inning",
    "balls_before",
    "strikes_before",
    "outs_before",
    "run_top_before",
    "run_bot_before",
    "run_total_before",
    "score_diff_home",
    "score_diff_pitcher_team",
    "runner_on_1b",
    "runner_on_2b",
    "runner_on_3b",
    "num_runners_on",
    "home_win_expectancy",
    "away_win_expectancy",
    "li",
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
    "asof_pitcher_pitchmix_n",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
)


RAW_CATEGORICAL = (
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_hand",
    "batter_hand",
)


IDENTITY_CATEGORICAL = (
    "pitcher_id",
    "batter_id",
    "pitcher_team_id",
    "batter_team_id",
)


CONTEXT_NUMERIC = (
    "ctx_pitcher_ahead",
    "ctx_batter_ahead",
    "ctx_two_strike",
    "ctx_three_ball",
    "ctx_full_count",
    "ctx_first_pitch",
    "ctx_risp",
    "ctx_bases_loaded",
    "ctx_force_length",
    "ctx_walk_forces_run",
    "ctx_double_play",
    "ctx_sac_fly",
    "ctx_pitch_around",
    "ctx_score_abs",
    "ctx_score_close",
    "ctx_score_tied",
    "ctx_late",
    "ctx_extra",
    "ctx_li_log1p",
    "ctx_high_leverage",
    "ctx_pressure",
    "ctx_must_zone",
    "ctx_waste_permission",
)


LATENT_NUMERIC = (
    "latent_history_reliability",
    "latent_recent_reliability",
    "latent_success_prior",
    "latent_recent_success",
    "latent_success_drift",
    "latent_middle_prior",
    "latent_recent_middle",
    "latent_heart_risk",
    "latent_waste_risk",
    "latent_wrong_side_risk",
    "latent_risk_union",
    "latent_command_margin",
    "latent_heart_x_must_zone",
    "latent_waste_x_permission",
    "latent_wrong_x_hand",
    "latent_mix_entropy",
    "latent_mix_concentration",
    "latent_mix_margin",
    "latent_expected_fastball",
    "latent_expected_breaking",
    "latent_expected_offspeed",
)


CONTEXT_CATEGORICAL = (
    "ctx_count_12",
    "ctx_base_out_24",
    "ctx_count_x_baseout",
    "ctx_score_bucket",
    "ctx_inning_bucket",
    "ctx_leverage_bucket",
    "ctx_hand_pair",
    "ctx_count_x_score_x_inning_x_li",
    "ctx_count_x_baseout_x_score_x_hand_x_game_type",
)


LATENT_CATEGORICAL = (
    "latent_expected_pitch_family",
    "latent_expected_x_count_x_hand",
    "latent_expected_x_baseout",
    "latent_heart_bucket",
    "latent_waste_bucket",
    "latent_wrong_side_bucket",
    "latent_risk_profile",
    "latent_risk_x_context",
    "latent_intent_side_pressure_proxy",
)


def _safe_numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).to_numpy(np.float64)


def _bucket(values: np.ndarray, edges: Iterable[float]) -> np.ndarray:
    return np.digitize(values, np.asarray(tuple(edges), dtype=np.float64), right=False)


def engineer_domain_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Pure row transform: no fit state and no reference to peer rows."""

    balls = _safe_numeric(raw, "balls_before").astype(np.int8)
    strikes = _safe_numeric(raw, "strikes_before").astype(np.int8)
    outs = _safe_numeric(raw, "outs_before").astype(np.int8)
    inning = _safe_numeric(raw, "inning").astype(np.int16)
    r1 = _safe_numeric(raw, "runner_on_1b").astype(np.int8)
    r2 = _safe_numeric(raw, "runner_on_2b").astype(np.int8)
    r3 = _safe_numeric(raw, "runner_on_3b").astype(np.int8)
    score = _safe_numeric(raw, "score_diff_pitcher_team")
    leverage = np.clip(_safe_numeric(raw, "li"), 0.0, 20.0)

    count = np.char.add(np.char.add(balls.astype(str), "-"), strikes.astype(str))
    base = raw["base_state"].astype("string").fillna("UNK").to_numpy(str)
    base_out = np.char.add(np.char.add(base, "|"), outs.astype(str))
    score_bucket_code = np.select(
        [score <= -4, score <= -2, score == -1, score == 0, score == 1, score <= 3],
        ["trail4+", "trail2-3", "trail1", "tie", "lead1", "lead2-3"],
        default="lead4+",
    )
    inning_bucket_code = np.select(
        [inning <= 3, inning <= 6, inning <= 9],
        ["early", "middle", "late"],
        default="extra",
    )
    leverage_bucket_code = np.select(
        [leverage < 0.7, leverage < 1.5, leverage < 2.5],
        ["low", "normal", "high"],
        default="extreme",
    )
    ph = raw["pitcher_hand"].astype("string").fillna("UNK").to_numpy(str)
    bh = raw["batter_hand"].astype("string").fillna("UNK").to_numpy(str)
    hand_pair = np.char.add(np.char.add(ph, "v"), bh)
    game_type = raw["game_type"].astype("string").fillna("UNK").to_numpy(str)

    force_length = r1 + (r1 & r2) + (r1 & r2 & r3)
    risp = (r2 | r3).astype(np.float64)
    bases_loaded = (r1 & r2 & r3).astype(np.float64)
    first_open = r1 == 0
    score_close = np.abs(score) <= 2
    high_leverage = leverage >= 1.5
    must_zone = balls == 3
    waste_permission = (strikes == 2) & (balls <= 1) & (bases_loaded == 0)

    pitcher_n = np.maximum(_safe_numeric(raw, "asof_pitcher_n"), 0.0)
    recent_available = np.zeros(len(raw), dtype=np.float64)

    def rate(column: str, fallback: float) -> np.ndarray:
        values = pd.to_numeric(raw[column], errors="coerce").to_numpy(np.float64)
        return np.where(np.isfinite(values), values, fallback)

    success = rate("asof_pitcher_success_rate", 0.5)
    reverse = rate("asof_pitcher_reverse_rate", 0.25)
    middle = rate("asof_pitcher_middle_rate", 0.15)
    ball_rate = rate("asof_pitcher_ball_rate", 0.40)
    batter_success = rate("asof_batter_success_rate", 0.5)
    batter_middle = rate("asof_batter_middle_rate", 0.15)

    recent_success_values: list[np.ndarray] = []
    recent_middle_values: list[np.ndarray] = []
    for horizon in (1, 3, 5):
        s = pd.to_numeric(
            raw[f"asof_pitcher_prev{horizon}_game_success_rate"], errors="coerce"
        ).to_numpy(np.float64)
        m = pd.to_numeric(
            raw[f"asof_pitcher_prev{horizon}_game_middle_rate"], errors="coerce"
        ).to_numpy(np.float64)
        recent_success_values.append(s)
        recent_middle_values.append(m)
        recent_available += np.isfinite(s).astype(np.float64)

    success_stack = np.vstack(recent_success_values)
    middle_stack = np.vstack(recent_middle_values)
    valid_success = np.isfinite(success_stack)
    valid_middle = np.isfinite(middle_stack)
    success_sum = np.where(valid_success, success_stack, 0.0).sum(axis=0)
    middle_sum = np.where(valid_middle, middle_stack, 0.0).sum(axis=0)
    success_count = valid_success.sum(axis=0)
    middle_count = valid_middle.sum(axis=0)
    recent_success = np.where(success_count > 0, success_sum / np.maximum(success_count, 1), success)
    recent_middle = np.where(middle_count > 0, middle_sum / np.maximum(middle_count, 1), middle)

    history_reliability = pitcher_n / (pitcher_n + 200.0)
    recent_reliability = recent_available / 3.0
    success_prior = history_reliability * success + (1.0 - history_reliability) * 0.5
    middle_prior = history_reliability * middle + (1.0 - history_reliability) * 0.15

    fastball = rate("asof_pitcher_fastball_rate", 1.0 / 3.0)
    breaking = rate("asof_pitcher_breaking_rate", 1.0 / 3.0)
    offspeed = rate("asof_pitcher_offspeed_rate", 1.0 / 3.0)
    mix = np.clip(np.vstack([fastball, breaking, offspeed]).T, 0.0, 1.0)
    mix_sum = mix.sum(axis=1, keepdims=True)
    mix = np.where(mix_sum > 1e-8, mix / mix_sum, 1.0 / 3.0)
    expected_index = np.argmax(mix, axis=1)
    expected_family = np.asarray(["fastball", "breaking", "offspeed"], dtype=object)[expected_index]
    sorted_mix = np.sort(mix, axis=1)
    mix_margin = sorted_mix[:, 2] - sorted_mix[:, 1]
    mix_concentration = np.sum(np.square(mix), axis=1)
    mix_entropy = -np.sum(mix * np.log(np.clip(mix, 1e-8, 1.0)), axis=1) / np.log(3.0)

    heart_risk = np.clip(
        0.50 * middle_prior
        + 0.25 * recent_middle
        + 0.15 * batter_middle
        + 0.10 * (leverage / (leverage + 2.0)),
        0.0,
        1.0,
    )
    waste_risk = np.clip(
        0.55 * ball_rate
        + 0.25 * waste_permission.astype(np.float64)
        + 0.10 * mix_entropy
        + 0.10 * (1.0 - success_prior),
        0.0,
        1.0,
    )
    same_hand = (ph == bh).astype(np.float64)
    wrong_side_risk = np.clip(
        0.65 * reverse
        + 0.15 * same_hand
        + 0.10 * mix_margin
        + 0.10 * (1.0 - success_prior),
        0.0,
        1.0,
    )
    # A representation only: subtype events overlap, so this is not treated as
    # a pseudo-label or literal decomposition of the binary target.
    risk_union = 1.0 - (1.0 - heart_risk) * (1.0 - waste_risk) * (1.0 - wrong_side_risk)
    command_margin = success_prior - 0.50 * risk_union + 0.10 * batter_success

    heart_bucket = _bucket(heart_risk, (0.15, 0.20, 0.25, 0.30)).astype(str)
    waste_bucket = _bucket(waste_risk, (0.35, 0.45, 0.55, 0.65)).astype(str)
    wrong_bucket = _bucket(wrong_side_risk, (0.20, 0.30, 0.40, 0.50)).astype(str)

    out = pd.DataFrame(index=raw.index)
    for column in RAW_NUMERIC:
        out[column] = pd.to_numeric(raw[column], errors="coerce").astype(np.float32)
    for column in RAW_CATEGORICAL + IDENTITY_CATEGORICAL:
        out[column] = raw[column].astype("string").fillna("UNK").astype(str)

    context_numeric = {
        "ctx_pitcher_ahead": strikes > balls,
        "ctx_batter_ahead": balls > strikes,
        "ctx_two_strike": strikes == 2,
        "ctx_three_ball": balls == 3,
        "ctx_full_count": (balls == 3) & (strikes == 2),
        "ctx_first_pitch": (balls == 0) & (strikes == 0),
        "ctx_risp": risp,
        "ctx_bases_loaded": bases_loaded,
        "ctx_force_length": force_length,
        "ctx_walk_forces_run": bases_loaded,
        "ctx_double_play": (r1 == 1) & (outs < 2),
        "ctx_sac_fly": (r3 == 1) & (outs < 2),
        "ctx_pitch_around": first_open & ((r2 == 1) | (r3 == 1)) & (inning >= 7) & score_close,
        "ctx_score_abs": np.abs(score),
        "ctx_score_close": score_close,
        "ctx_score_tied": score == 0,
        "ctx_late": inning >= 7,
        "ctx_extra": inning >= 10,
        "ctx_li_log1p": np.log1p(leverage),
        "ctx_high_leverage": high_leverage,
        "ctx_pressure": np.log1p(leverage) * (1.0 + 0.5 * score_close + 0.5 * risp) * (1.0 + 0.5 * must_zone),
        "ctx_must_zone": must_zone,
        "ctx_waste_permission": waste_permission,
    }
    for column, values in context_numeric.items():
        out[column] = np.asarray(values, dtype=np.float32)

    latent_numeric = {
        "latent_history_reliability": history_reliability,
        "latent_recent_reliability": recent_reliability,
        "latent_success_prior": success_prior,
        "latent_recent_success": recent_success,
        "latent_success_drift": recent_success - success_prior,
        "latent_middle_prior": middle_prior,
        "latent_recent_middle": recent_middle,
        "latent_heart_risk": heart_risk,
        "latent_waste_risk": waste_risk,
        "latent_wrong_side_risk": wrong_side_risk,
        "latent_risk_union": risk_union,
        "latent_command_margin": command_margin,
        "latent_heart_x_must_zone": heart_risk * (1.0 + must_zone + 0.5 * high_leverage),
        "latent_waste_x_permission": waste_risk * (1.0 + waste_permission),
        "latent_wrong_x_hand": wrong_side_risk * (1.0 + same_hand),
        "latent_mix_entropy": mix_entropy,
        "latent_mix_concentration": mix_concentration,
        "latent_mix_margin": mix_margin,
        "latent_expected_fastball": expected_index == 0,
        "latent_expected_breaking": expected_index == 1,
        "latent_expected_offspeed": expected_index == 2,
    }
    for column, values in latent_numeric.items():
        out[column] = np.asarray(values, dtype=np.float32)

    count_x_baseout = np.char.add(np.char.add(count, "|"), base_out)
    pressure_token = np.char.add(
        np.char.add(
            np.char.add(np.char.add(count, "|"), score_bucket_code.astype(str)),
            np.char.add("|", inning_bucket_code.astype(str)),
        ),
        np.char.add("|", leverage_bucket_code.astype(str)),
    )
    tactical = count_x_baseout
    for values in (score_bucket_code, hand_pair, game_type):
        tactical = np.char.add(np.char.add(tactical, "|"), values.astype(str))

    context_categories = {
        "ctx_count_12": count,
        "ctx_base_out_24": base_out,
        "ctx_count_x_baseout": count_x_baseout,
        "ctx_score_bucket": score_bucket_code,
        "ctx_inning_bucket": inning_bucket_code,
        "ctx_leverage_bucket": leverage_bucket_code,
        "ctx_hand_pair": hand_pair,
        "ctx_count_x_score_x_inning_x_li": pressure_token,
        "ctx_count_x_baseout_x_score_x_hand_x_game_type": tactical,
    }
    for column, values in context_categories.items():
        out[column] = pd.Series(values, index=raw.index, dtype="string").fillna("UNK").astype(str)

    expected_x_count_x_hand = expected_family.astype(str)
    for values in (count, hand_pair):
        expected_x_count_x_hand = np.char.add(
            np.char.add(expected_x_count_x_hand, "|"), values.astype(str)
        )
    expected_x_baseout = np.char.add(
        np.char.add(expected_family.astype(str), "|"), base_out.astype(str)
    )
    risk_profile = heart_bucket
    for values in (waste_bucket, wrong_bucket):
        risk_profile = np.char.add(np.char.add(risk_profile, "|"), values)
    risk_x_context = np.char.add(np.char.add(risk_profile, "|"), count_x_baseout)
    side_proxy = expected_family.astype(str)
    for values in (hand_pair, wrong_bucket, count):
        side_proxy = np.char.add(np.char.add(side_proxy, "|"), values.astype(str))

    latent_categories = {
        "latent_expected_pitch_family": expected_family,
        "latent_expected_x_count_x_hand": expected_x_count_x_hand,
        "latent_expected_x_baseout": expected_x_baseout,
        "latent_heart_bucket": heart_bucket,
        "latent_waste_bucket": waste_bucket,
        "latent_wrong_side_bucket": wrong_bucket,
        "latent_risk_profile": risk_profile,
        "latent_risk_x_context": risk_x_context,
        "latent_intent_side_pressure_proxy": side_proxy,
    }
    for column, values in latent_categories.items():
        out[column] = pd.Series(values, index=raw.index, dtype="string").fillna("UNK").astype(str)
    return out.reset_index(drop=True)


def add_intent_features(rows: pd.DataFrame) -> pd.DataFrame:
    """Intent branch의 공개 행 단위 API."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
        return engineer_domain_features(rows)

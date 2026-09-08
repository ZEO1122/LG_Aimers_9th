"""Prior-season TrackMan alignment, aggregation, and frozen-row joins."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

PRIMARY_HISTORY_COLUMNS = (
    "tm_pitcher_recent1_rel_speed_median",
    "tm_pitcher_recent1_rel_speed_mean",
    "tm_pitcher_career_horz_break_std",
    "tm_pitcher_career_type_breaking_spin_rate_std",
    "tm_pitcher_career_type_offspeed_horz_break_std",
    "tm_pitcher_recent1_pitch_group_offspeed_rate",
    "tm_pitcher_recent2_zone_speed_median",
    "tm_pitcher_career_pitch_group_offspeed_rate",
    "tm_pitcher_career_type_offspeed_spin_rate_std",
    "tm_pitcher_career_type_fastball_horz_break_mean",
    "tm_pitcher_career_type_other_induced_vert_break_mean",
    "tm_pitcher_recent1_induced_vert_break_std",
    "tm_pitcher_career_type_breaking_rel_side_mean",
    "tm_pitcher_recent1_movement_magnitude_mean",
    "tm_pitcher_recent1_movement_magnitude_std",
    "tm_pitcher_career_type_breaking_horz_break_mean",
    "tm_pitcher_career_movement_magnitude_mean",
    "tm_pitcher_career_type_offspeed_rel_speed_mean",
    "tm_pitcher_career_type_other_n",
    "tm_pitcher_career_pitch_group_other_rate",
    "tm_pitcher_recent1_pitch_group_other_rate",
    "tm_pitcher_career_type_offspeed_horz_break_mean",
    "tm_pitcher_recent1_spin_rate_mean",
    "tm_pitcher_recent1_spin_rate_median",
    "tm_pitcher_recent1_movement_magnitude_median",
    "tm_pitcher_recent1_rel_height_mean",
    "tm_pitcher_recent1_velo_loss_median",
    "tm_pitcher_career_type_breaking_induced_vert_break_mean",
)

SECONDARY_HISTORY_COLUMNS = (
    "tm_pitcher_recent2_horz_break_std",
    "tm_pitcher_career_type_offspeed_horz_break_std",
    "tm_pitcher_recent1_pitch_group_offspeed_rate",
    "tm_pitcher_career_horz_break_std",
    "tm_pitcher_recent2_pitch_group_offspeed_rate",
    "tm_pitcher_recent2_horz_break_median",
    "tm_pitcher_recent1_horz_break_std",
    "tm_pitcher_career_type_fastball_horz_break_mean",
    "tm_pitcher_recent1_horz_break_median",
    "tm_pitcher_recent2_rel_side_mean",
    "tm_pitcher_recent2_rel_speed_median",
    "tm_pitcher_recent2_induced_vert_break_std",
    "tm_pitcher_career_type_breaking_spin_rate_std",
    "tm_pitcher_career_pitch_group_offspeed_rate",
    "tm_pitcher_career_induced_vert_break_std",
    "tm_pitcher_career_type_breaking_horz_break_mean",
    "tm_pitcher_recent1_movement_magnitude_std",
    "tm_pitcher_recent1_induced_vert_break_std",
    "tm_pitcher_recent1_rel_speed_median",
    "tm_pitcher_career_type_breaking_induced_vert_break_mean",
    "tm_pitcher_recent1_movement_magnitude_mean",
)

HISTORY_UNION_COLUMNS = tuple(dict.fromkeys(PRIMARY_HISTORY_COLUMNS + SECONDARY_HISTORY_COLUMNS))

Role = Literal["pitcher", "batter"]

MAIN_ALIGNMENT_COLUMNS: tuple[str, ...] = (
    "season",
    "inning",
    "top_bottom",
    "balls_before",
    "strikes_before",
    "outs_before",
    "pitcher_id",
    "batter_id",
    "pitcher_hand",
    "batter_hand",
)

HISTORY_ALIGNMENT_COLUMNS: tuple[str, ...] = (
    "season",
    "trackman_game_id",
    "pitch_no",
    "inning",
    "top_bottom",
    "balls_before",
    "strikes_before",
    "outs_before",
    "pitcher_trackman_id",
    "batter_trackman_id",
    "pitcher_hand",
    "batter_hand",
)

PHYSICAL_COLUMNS: tuple[str, ...] = (
    "rel_speed",
    "zone_speed",
    "spin_rate",
    "induced_vert_break",
    "horz_break",
    "extension",
    "rel_height",
    "rel_side",
)

HISTORY_FEATURE_COLUMNS: tuple[str, ...] = (
    "season",
    "game_date",
    "trackman_game_id",
    "pitch_no",
    "pitcher_trackman_id",
    "batter_trackman_id",
    "tagged_pitch_type",
    "auto_pitch_type",
    "pitch_type_group",
    *PHYSICAL_COLUMNS,
)

_ROLE_COLUMNS: Mapping[Role, tuple[str, str, str, str]] = {
    "pitcher": (
        "pitcher_id",
        "pitcher_trackman_id",
        "pitcher_hand",
        "pitcher_hand",
    ),
    "batter": (
        "batter_id",
        "batter_trackman_id",
        "batter_hand",
        "batter_hand",
    ),
}

_HISTORY_HAND_MAP = {"Left": 1, "Right": 2}
_PITCH_GROUPS = ("fastball", "breaking", "offspeed", "other")


@dataclass(frozen=True)
class HistoryFeatureBundle:
    """All auditable outputs from :func:`build_history_feature_bundle`."""

    pitcher_features: pd.DataFrame
    batter_features: pd.DataFrame
    entity_mapping: pd.DataFrame
    mapping_evidence: pd.DataFrame
    game_matches: pd.DataFrame
    diagnostics: dict[str, object]


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _state_tokens(frame: pd.DataFrame, *, main: bool) -> np.ndarray:
    half_map = {"T": 0, "B": 1} if main else {"Top": 0, "Bottom": 1}
    half = frame["top_bottom"].map(half_map)
    if half.isna().any():
        bad = frame.loc[half.isna(), "top_bottom"].drop_duplicates().tolist()
        raise ValueError(f"unexpected top_bottom values: {bad}")

    # Maximum legal value is comfortably below int16's 32,767 limit.
    token = (
        frame["inning"].to_numpy(dtype=np.int16) * 1_000
        + half.to_numpy(dtype=np.int16) * 500
        + frame["outs_before"].to_numpy(dtype=np.int16) * 100
        + frame["balls_before"].to_numpy(dtype=np.int16) * 10
        + frame["strikes_before"].to_numpy(dtype=np.int16)
    )
    return token.astype("<i2", copy=False)


def _digest(values: np.ndarray) -> str:
    return hashlib.blake2b(values.tobytes(), digest_size=12).hexdigest()


def _prepare_main_games(
    main: pd.DataFrame, prefix_len: int
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    _require_columns(main, MAIN_ALIGNMENT_COLUMNS, "main")
    if prefix_len < 8:
        raise ValueError("prefix_len must be at least 8 pre-pitch states")

    ordered = main.loc[:, MAIN_ALIGNMENT_COLUMNS].reset_index(drop=True)
    if not ordered["season"].is_monotonic_increasing:
        raise ValueError(
            "main rows must retain their original chronological order; season is not monotonic"
        )

    new_game = ordered["season"].ne(ordered["season"].shift()) | ordered[
        "inning"
    ].lt(ordered["inning"].shift())
    new_game.iloc[0] = True
    starts = np.flatnonzero(new_game.to_numpy())
    stops = np.r_[starts[1:], len(ordered)]
    tokens = _state_tokens(ordered, main=True)

    records: list[dict[str, object]] = []
    seasons = ordered["season"].to_numpy()
    for game_index, (start, stop) in enumerate(zip(starts, stops, strict=True)):
        length = int(stop - start)
        records.append(
            {
                "main_game_index": game_index,
                "season": int(seasons[start]),
                "main_start": int(start),
                "main_stop": int(stop),
                "main_length": length,
                "full_digest": _digest(tokens[start:stop]),
                "prefix_digest": (
                    _digest(tokens[start : start + prefix_len])
                    if length >= prefix_len
                    else None
                ),
            }
        )
    return ordered, pd.DataFrame.from_records(records), tokens


def _prepare_history_games(
    history: pd.DataFrame, prefix_len: int
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, int]:
    _require_columns(history, HISTORY_ALIGNMENT_COLUMNS, "history")
    selected = history.loc[:, HISTORY_ALIGNMENT_COLUMNS].copy()
    selected.sort_values(
        ["season", "trackman_game_id", "pitch_no"], kind="stable", inplace=True
    )
    duplicate_pitch_rows = int(
        selected.duplicated(["season", "trackman_game_id", "pitch_no"]).sum()
    )
    # Duplicate game/pitch keys are data artifacts, not additional observations.
    selected.drop_duplicates(
        ["season", "trackman_game_id", "pitch_no"], keep="first", inplace=True
    )
    selected.reset_index(drop=True, inplace=True)

    new_game = selected["season"].ne(selected["season"].shift()) | selected[
        "trackman_game_id"
    ].ne(selected["trackman_game_id"].shift())
    new_game.iloc[0] = True
    starts = np.flatnonzero(new_game.to_numpy())
    stops = np.r_[starts[1:], len(selected)]
    tokens = _state_tokens(selected, main=False)

    records: list[dict[str, object]] = []
    seasons = selected["season"].to_numpy()
    game_ids = selected["trackman_game_id"].to_numpy()
    for game_index, (start, stop) in enumerate(zip(starts, stops, strict=True)):
        length = int(stop - start)
        records.append(
            {
                "history_game_index": game_index,
                "season": int(seasons[start]),
                "history_start": int(start),
                "history_stop": int(stop),
                "history_length": length,
                "trackman_game_id": game_ids[start],
                "full_digest": _digest(tokens[start:stop]),
                "prefix_digest": (
                    _digest(tokens[start : start + prefix_len])
                    if length >= prefix_len
                    else None
                ),
            }
        )
    return selected, pd.DataFrame.from_records(records), tokens, duplicate_pitch_rows


def _unique_rows(frame: pd.DataFrame, subset: Sequence[str]) -> pd.DataFrame:
    return frame.loc[~frame.duplicated(list(subset), keep=False)].copy()


def _match_games(
    main_games: pd.DataFrame, history_games: pd.DataFrame, prefix_len: int
) -> pd.DataFrame:
    exact_keys = ["season", "main_length", "full_digest"]
    left = main_games.rename(columns={"main_length": "match_length"})
    right = history_games.rename(columns={"history_length": "match_length"})
    exact_subset = ["season", "match_length", "full_digest"]
    exact_left = _unique_rows(left, exact_subset)
    exact_right = _unique_rows(right, exact_subset)
    exact = exact_left.merge(
        exact_right,
        on=exact_subset,
        how="inner",
        validate="one_to_one",
        suffixes=("_main", "_history"),
    )
    exact_out = exact.loc[
        :,
        [
            "main_game_index",
            "history_game_index",
            "season",
            "main_start",
            "main_stop",
            "history_start",
            "history_stop",
            "match_length",
            "trackman_game_id",
        ],
    ].rename(columns={"match_length": "aligned_rows"})
    exact_out["match_method"] = "exact"
    exact_out["main_length"] = exact_out["main_stop"] - exact_out["main_start"]
    exact_out["history_length"] = (
        exact_out["history_stop"] - exact_out["history_start"]
    )

    used_main = set(exact_out["main_game_index"])
    used_history = set(exact_out["history_game_index"])
    prefix_left = main_games.loc[
        ~main_games["main_game_index"].isin(used_main)
        & main_games["prefix_digest"].notna()
    ]
    prefix_right = history_games.loc[
        ~history_games["history_game_index"].isin(used_history)
        & history_games["prefix_digest"].notna()
    ]
    prefix_subset = ["season", "prefix_digest"]
    prefix_left = _unique_rows(prefix_left, prefix_subset)
    prefix_right = _unique_rows(prefix_right, prefix_subset)
    prefix = prefix_left.merge(
        prefix_right,
        on=prefix_subset,
        how="inner",
        validate="one_to_one",
        suffixes=("_main", "_history"),
    )
    prefix_out = prefix.loc[
        :,
        [
            "main_game_index",
            "history_game_index",
            "season",
            "main_start",
            "main_stop",
            "history_start",
            "history_stop",
            "main_length",
            "history_length",
            "trackman_game_id",
        ],
    ].copy()
    prefix_out["aligned_rows"] = prefix_len
    prefix_out["match_method"] = "prefix"

    columns = [
        "main_game_index",
        "history_game_index",
        "season",
        "main_start",
        "main_stop",
        "history_start",
        "history_stop",
        "main_length",
        "history_length",
        "aligned_rows",
        "trackman_game_id",
        "match_method",
    ]
    matches = pd.concat([exact_out[columns], prefix_out[columns]], ignore_index=True)
    matches.sort_values(["season", "main_game_index"], inplace=True)
    matches.reset_index(drop=True, inplace=True)
    return matches


def _alignment_evidence_for_role(
    main: pd.DataFrame,
    history: pd.DataFrame,
    matches: pd.DataFrame,
    role: Role,
) -> tuple[pd.DataFrame, dict[str, object]]:
    main_id_col, history_id_col, main_hand_col, history_hand_col = _ROLE_COLUMNS[role]
    main_ids: list[np.ndarray] = []
    history_ids: list[np.ndarray] = []
    seasons: list[np.ndarray] = []
    match_indices: list[np.ndarray] = []
    is_exact: list[np.ndarray] = []
    total_aligned = 0
    hand_matched = 0

    history_hand = history[history_hand_col].map(_HISTORY_HAND_MAP)
    if history_hand.isna().any():
        bad = history.loc[history_hand.isna(), history_hand_col].drop_duplicates().tolist()
        raise ValueError(f"unexpected TrackMan hand values for {role}: {bad}")

    for match_index, row in enumerate(matches.itertuples(index=False)):
        count = int(row.aligned_rows)
        main_start = int(row.main_start)
        history_start = int(row.history_start)
        main_slice = slice(main_start, main_start + count)
        history_slice = slice(history_start, history_start + count)
        total_aligned += count

        hand_ok = (
            main[main_hand_col].iloc[main_slice].to_numpy()
            == history_hand.iloc[history_slice].to_numpy()
        )
        hand_matched += int(hand_ok.sum())
        if not hand_ok.any():
            continue

        main_ids.append(main[main_id_col].iloc[main_slice].to_numpy()[hand_ok])
        history_ids.append(
            history[history_id_col].iloc[history_slice].to_numpy()[hand_ok]
        )
        n_ok = int(hand_ok.sum())
        seasons.append(np.full(n_ok, int(row.season), dtype=np.int16))
        match_indices.append(np.full(n_ok, match_index, dtype=np.int32))
        is_exact.append(
            np.full(n_ok, row.match_method == "exact", dtype=np.int8)
        )

    if not main_ids:
        empty = pd.DataFrame(
            columns=[
                "role",
                "season",
                "main_entity_id",
                "trackman_entity_id",
                "votes",
                "evidence_games",
                "exact_votes",
                "prefix_votes",
            ]
        )
        return empty, {
            "aligned_rows": total_aligned,
            "hand_matched_rows": 0,
            "hand_agreement": 0.0,
        }

    raw = pd.DataFrame(
        {
            "season": np.concatenate(seasons),
            "main_entity_id": np.concatenate(main_ids),
            "trackman_entity_id": np.concatenate(history_ids),
            "match_index": np.concatenate(match_indices),
            "is_exact": np.concatenate(is_exact),
        }
    )
    evidence = (
        raw.groupby(
            ["season", "main_entity_id", "trackman_entity_id"], sort=False
        )
        .agg(
            votes=("main_entity_id", "size"),
            evidence_games=("match_index", "nunique"),
            exact_votes=("is_exact", "sum"),
        )
        .reset_index()
    )
    evidence["prefix_votes"] = evidence["votes"] - evidence["exact_votes"]
    evidence.insert(0, "role", role)
    diagnostics = {
        "aligned_rows": total_aligned,
        "hand_matched_rows": hand_matched,
        "hand_agreement": hand_matched / total_aligned if total_aligned else None,
        "evidence_pairs": int(len(evidence)),
        "main_entities_with_evidence": int(evidence["main_entity_id"].nunique()),
        "trackman_entities_with_evidence": int(
            evidence["trackman_entity_id"].nunique()
        ),
    }
    return evidence, diagnostics


def build_alignment_evidence(
    main: pd.DataFrame,
    history: pd.DataFrame,
    *,
    prefix_len: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Match historical games and return context-only entity-pair votes.

    The evidence covers every available public season.  A later call to
    :func:`infer_entity_mapping` applies the strict ``season < cutoff_year``
    filter.  No target column, pitch type, or physical measurement is inspected.
    """

    prepared_main, main_games, main_tokens = _prepare_main_games(main, prefix_len)
    prepared_history, history_games, history_tokens, duplicate_rows = (
        _prepare_history_games(history, prefix_len)
    )
    matches = _match_games(main_games, history_games, prefix_len)

    # Defense in depth: every aligned prefix must be byte-identical.
    for row in matches.itertuples(index=False):
        count = int(row.aligned_rows)
        left = main_tokens[int(row.main_start) : int(row.main_start) + count]
        right = history_tokens[
            int(row.history_start) : int(row.history_start) + count
        ]
        if not np.array_equal(left, right):
            raise AssertionError("a reported game match has unequal state tokens")

    role_outputs: list[pd.DataFrame] = []
    role_diagnostics: dict[str, object] = {}
    for role in ("pitcher", "batter"):
        evidence, diagnostics = _alignment_evidence_for_role(
            prepared_main, prepared_history, matches, role
        )
        role_outputs.append(evidence)
        role_diagnostics[role] = diagnostics

    evidence = pd.concat(role_outputs, ignore_index=True)
    exact = matches["match_method"].eq("exact")
    diagnostics = {
        "prefix_len": prefix_len,
        "main_games": int(len(main_games)),
        "history_games": int(len(history_games)),
        "matched_games": int(len(matches)),
        "exact_matched_games": int(exact.sum()),
        "prefix_matched_games": int((~exact).sum()),
        "exact_aligned_rows": int(matches.loc[exact, "aligned_rows"].sum()),
        "prefix_aligned_rows": int(matches.loc[~exact, "aligned_rows"].sum()),
        "history_duplicate_game_pitch_rows_removed": duplicate_rows,
        "roles": role_diagnostics,
    }
    return evidence, matches, diagnostics


def infer_entity_mapping(
    evidence: pd.DataFrame,
    *,
    cutoff_year: int,
    role: Role,
    min_votes: int = 20,
    min_purity: float = 0.99,
    require_reciprocal_best: bool = True,
) -> pd.DataFrame:
    """Infer a role-specific entity map using evidence strictly before Y."""

    if not 0.5 <= min_purity <= 1.0:
        raise ValueError("min_purity must lie in [0.5, 1.0]")
    if min_votes < 1:
        raise ValueError("min_votes must be positive")
    prior = evidence.loc[
        evidence["role"].eq(role) & evidence["season"].lt(cutoff_year)
    ].copy()
    if not prior.empty and int(prior["season"].max()) >= cutoff_year:
        raise AssertionError("lookahead evidence reached mapping inference")

    output_columns = [
        "cutoff_year",
        "role",
        "main_entity_id",
        "trackman_entity_id",
        "selected_votes",
        "total_votes",
        "candidate_count",
        "purity",
        "reciprocal_purity",
        "evidence_games",
        "exact_votes",
        "prefix_votes",
        "evidence_seasons",
        "last_evidence_season",
        "reciprocal_best",
        "accepted",
    ]
    if prior.empty:
        return pd.DataFrame(columns=output_columns)

    pair = (
        prior.groupby(["main_entity_id", "trackman_entity_id"], sort=False)
        .agg(
            selected_votes=("votes", "sum"),
            evidence_games=("evidence_games", "sum"),
            exact_votes=("exact_votes", "sum"),
            prefix_votes=("prefix_votes", "sum"),
        )
        .reset_index()
    )
    by_main = pair.groupby("main_entity_id", sort=False)
    pair["total_votes"] = by_main["selected_votes"].transform("sum")
    pair["candidate_count"] = by_main["trackman_entity_id"].transform("size")
    pair.sort_values(
        ["main_entity_id", "selected_votes", "trackman_entity_id"],
        ascending=[True, False, True],
        inplace=True,
    )
    selected = pair.drop_duplicates("main_entity_id", keep="first").copy()
    selected["purity"] = selected["selected_votes"] / selected["total_votes"]

    hist_totals = pair.groupby("trackman_entity_id")["selected_votes"].sum()
    reciprocal = pair.sort_values(
        ["trackman_entity_id", "selected_votes", "main_entity_id"],
        ascending=[True, False, True],
    ).drop_duplicates("trackman_entity_id", keep="first")
    reciprocal_top = reciprocal.set_index("trackman_entity_id")["main_entity_id"]
    selected["reciprocal_best"] = selected["main_entity_id"].eq(
        selected["trackman_entity_id"].map(reciprocal_top)
    )
    selected["reciprocal_purity"] = selected["selected_votes"] / selected[
        "trackman_entity_id"
    ].map(hist_totals)

    season_stats = prior.groupby("main_entity_id").agg(
        evidence_seasons=("season", "nunique"),
        last_evidence_season=("season", "max"),
    )
    selected = selected.merge(
        season_stats, left_on="main_entity_id", right_index=True, how="left"
    )
    accepted = selected["selected_votes"].ge(min_votes) & selected["purity"].ge(
        min_purity
    )
    if require_reciprocal_best:
        accepted &= selected["reciprocal_best"]
    selected["accepted"] = accepted
    selected.insert(0, "role", role)
    selected.insert(0, "cutoff_year", cutoff_year)
    return selected.loc[:, output_columns].reset_index(drop=True)


def _deduplicated_history(history: pd.DataFrame) -> pd.DataFrame:
    _require_columns(history, HISTORY_FEATURE_COLUMNS, "history")
    frame = history.loc[:, HISTORY_FEATURE_COLUMNS].copy()
    frame.sort_values(
        ["season", "trackman_game_id", "pitch_no"], kind="stable", inplace=True
    )
    frame.drop_duplicates(
        ["season", "trackman_game_id", "pitch_no"], keep="first", inplace=True
    )
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
    frame["velo_loss"] = frame["rel_speed"] - frame["zone_speed"]
    frame["movement_magnitude"] = np.hypot(
        frame["induced_vert_break"], frame["horz_break"]
    )
    frame["release_distance"] = np.hypot(frame["rel_height"], frame["rel_side"])
    frame["tag_auto_agree"] = frame["tagged_pitch_type"].eq(
        frame["auto_pitch_type"]
    )
    return frame


_AGG_METRICS: tuple[str, ...] = (
    *PHYSICAL_COLUMNS,
    "velo_loss",
    "movement_magnitude",
    "release_distance",
)

_TYPE_METRICS: tuple[str, ...] = (
    "rel_speed",
    "spin_rate",
    "induced_vert_break",
    "horz_break",
    "extension",
    "rel_height",
    "rel_side",
)


def _window_aggregates(
    frame: pd.DataFrame,
    *,
    key: str,
    prefix: str,
    cutoff_year: int,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(index=pd.Index([], name=key))
    grouped = frame.groupby(key, sort=False, observed=True)
    base = grouped.agg(
        n_pitches=("pitch_no", "size"),
        n_games=("trackman_game_id", "nunique"),
        n_seasons=("season", "nunique"),
        first_season=("season", "min"),
        last_season=("season", "max"),
        first_date=("game_date", "min"),
        last_date=("game_date", "max"),
        tagged_auto_agree_rate=("tag_auto_agree", "mean"),
    )
    cutoff_date = pd.Timestamp(year=cutoff_year, month=1, day=1)
    base["career_span_days"] = (base["last_date"] - base["first_date"]).dt.days
    base["days_since_last"] = (cutoff_date - base["last_date"]).dt.days
    base.drop(columns=["first_date", "last_date"], inplace=True)

    metric = grouped[list(_AGG_METRICS)].agg(["mean", "std", "median"])
    metric.columns = [f"{column}_{stat}" for column, stat in metric.columns]
    availability = grouped[list(_AGG_METRICS)].count().div(base["n_pitches"], axis=0)
    availability.columns = [f"{column}_availability" for column in availability]

    pitch_counts = pd.crosstab(frame[key], frame["pitch_type_group"])
    pitch_counts = pitch_counts.reindex(columns=_PITCH_GROUPS, fill_value=0)
    pitch_rates = pitch_counts.div(pitch_counts.sum(axis=1), axis=0)
    pitch_rates.columns = [f"pitch_group_{column}_rate" for column in pitch_rates]

    output = base.join(metric).join(availability).join(pitch_rates)
    output.columns = [prefix + column for column in output.columns]
    return output


def _type_specific_aggregates(
    frame: pd.DataFrame, *, key: str, prefix: str
) -> pd.DataFrame:
    usable = frame.loc[frame["pitch_type_group"].isin(_PITCH_GROUPS)]
    if usable.empty:
        return pd.DataFrame(index=pd.Index([], name=key))
    grouped = usable.groupby([key, "pitch_type_group"], observed=True, sort=False)
    counts = grouped.size().unstack("pitch_type_group")
    counts = counts.reindex(columns=_PITCH_GROUPS, fill_value=0)
    counts.columns = [f"{prefix}{group}_n" for group in counts.columns]

    metrics = grouped[list(_TYPE_METRICS)].agg(["mean", "std"])
    wide = metrics.unstack("pitch_type_group")
    wide = wide.reindex(columns=_PITCH_GROUPS, level=2)
    wide.columns = [
        f"{prefix}{group}_{metric}_{stat}" for metric, stat, group in wide.columns
    ]
    return counts.join(wide)


def aggregate_history_features(
    history: pd.DataFrame,
    *,
    cutoff_year: int,
    role: Role,
    recent_windows: Sequence[int] = (1, 2),
    include_type_specific: bool = True,
) -> pd.DataFrame:
    """Aggregate only TrackMan seasons strictly before ``cutoff_year``.

    ``recent_windows=(1, 2)`` adds the immediately preceding season and the
    preceding two-season window alongside career-to-date aggregates.
    """

    if any(window < 1 for window in recent_windows):
        raise ValueError("recent windows must be positive numbers of seasons")
    key = _ROLE_COLUMNS[role][1]
    prepared = _deduplicated_history(history)
    prior = prepared.loc[prepared["season"].lt(cutoff_year)].copy()
    if prior.empty:
        return pd.DataFrame(columns=[key])
    if int(prior["season"].max()) >= cutoff_year:
        raise AssertionError("history lookahead reached aggregate construction")

    prefix_root = f"tm_{role}_"
    parts = [
        _window_aggregates(
            prior,
            key=key,
            prefix=prefix_root + "career_",
            cutoff_year=cutoff_year,
        )
    ]
    for window in sorted(set(recent_windows)):
        recent = prior.loc[prior["season"].ge(cutoff_year - window)]
        parts.append(
            _window_aggregates(
                recent,
                key=key,
                prefix=prefix_root + f"recent{window}_",
                cutoff_year=cutoff_year,
            )
        )
    if include_type_specific:
        parts.append(
            _type_specific_aggregates(
                prior, key=key, prefix=prefix_root + "career_type_"
            )
        )
    output = parts[0]
    for part in parts[1:]:
        output = output.join(part, how="left")
    return output.reset_index()


def _attach_aggregates(
    mapping: pd.DataFrame,
    aggregates: pd.DataFrame,
    *,
    role: Role,
) -> pd.DataFrame:
    main_id_col, history_id_col, _, _ = _ROLE_COLUMNS[role]
    accepted = mapping.loc[mapping["accepted"]].copy()
    if accepted.empty:
        return pd.DataFrame(columns=["cutoff_year", main_id_col])
    accepted.rename(
        columns={
            "main_entity_id": main_id_col,
            "trackman_entity_id": f"tm_{role}_id",
            "selected_votes": f"tm_{role}_map_votes",
            "purity": f"tm_{role}_map_purity",
            "reciprocal_purity": f"tm_{role}_map_reciprocal_purity",
            "evidence_games": f"tm_{role}_map_games",
            "evidence_seasons": f"tm_{role}_map_seasons",
            "last_evidence_season": f"tm_{role}_map_last_season",
        },
        inplace=True,
    )
    keep = [
        "cutoff_year",
        main_id_col,
        f"tm_{role}_id",
        f"tm_{role}_map_votes",
        f"tm_{role}_map_purity",
        f"tm_{role}_map_reciprocal_purity",
        f"tm_{role}_map_games",
        f"tm_{role}_map_seasons",
        f"tm_{role}_map_last_season",
    ]
    output = accepted.loc[:, keep].merge(
        aggregates,
        left_on=f"tm_{role}_id",
        right_on=history_id_col,
        how="left",
        validate="one_to_one",
    )
    output.drop(columns=[history_id_col], inplace=True)
    return output


def _mapping_diagnostics(
    mapping: pd.DataFrame,
    main: pd.DataFrame,
    *,
    cutoff_year: int,
    role: Role,
) -> dict[str, object]:
    main_id_col = _ROLE_COLUMNS[role][0]
    prior_ids = main.loc[main["season"].lt(cutoff_year), main_id_col]
    accepted = mapping.loc[mapping["accepted"]]
    accepted_ids = set(accepted["main_entity_id"])
    diagnostic: dict[str, object] = {
        "source_entities": int(prior_ids.nunique()),
        "entities_with_candidate": int(len(mapping)),
        "accepted_entities": int(len(accepted)),
        "source_row_coverage": float(prior_ids.isin(accepted_ids).mean())
        if len(prior_ids)
        else None,
        "accepted_min_purity": float(accepted["purity"].min())
        if len(accepted)
        else None,
        "accepted_weighted_purity": float(
            np.average(accepted["purity"], weights=accepted["selected_votes"])
        )
        if len(accepted)
        else None,
    }
    # Validation coverage is safe to compute for public train years.  Test rows
    # are deliberately never passed to this module or inspected here.
    target = main.loc[main["season"].eq(cutoff_year), main_id_col]
    if len(target):
        diagnostic["public_validation_rows"] = int(len(target))
        diagnostic["public_validation_row_coverage"] = float(
            target.isin(accepted_ids).mean()
        )
        diagnostic["public_validation_entity_coverage"] = float(
            pd.Index(target.unique()).isin(accepted_ids).mean()
        )
    else:
        diagnostic["public_validation_rows"] = 0
        diagnostic["public_validation_row_coverage"] = None
        diagnostic["public_validation_entity_coverage"] = None
    return diagnostic


def build_history_feature_bundle(
    main: pd.DataFrame,
    history: pd.DataFrame,
    *,
    cutoff_years: Sequence[int],
    prefix_len: int = 50,
    min_votes: int = 20,
    min_purity: float = 0.99,
    recent_windows: Sequence[int] = (1, 2),
    include_type_specific: bool = True,
) -> HistoryFeatureBundle:
    """Build leakage-safe mappings/features for one or more cutoff years."""

    if not cutoff_years:
        raise ValueError("at least one cutoff year is required")
    cutoffs = sorted(set(map(int, cutoff_years)))
    evidence, matches, alignment_diagnostics = build_alignment_evidence(
        main, history, prefix_len=prefix_len
    )

    pitcher_outputs: list[pd.DataFrame] = []
    batter_outputs: list[pd.DataFrame] = []
    mappings: list[pd.DataFrame] = []
    cutoff_diagnostics: dict[str, object] = {}
    history_min_season = int(history["season"].min())
    history_max_season = int(history["season"].max())

    for cutoff_year in cutoffs:
        if cutoff_year <= history_min_season:
            raise ValueError(
                f"cutoff_year={cutoff_year} has no prior TrackMan season"
            )
        cutoff_info: dict[str, object] = {
            "main_max_season_used": int(
                main.loc[main["season"].lt(cutoff_year), "season"].max()
            ),
            "history_max_season_used": min(history_max_season, cutoff_year - 1),
            "test_rows_inspected": 0,
        }
        if cutoff_info["main_max_season_used"] >= cutoff_year:
            raise AssertionError("main lookahead guard failed")
        if cutoff_info["history_max_season_used"] >= cutoff_year:
            raise AssertionError("history lookahead guard failed")

        role_info: dict[str, object] = {}
        for role in ("pitcher", "batter"):
            mapping = infer_entity_mapping(
                evidence,
                cutoff_year=cutoff_year,
                role=role,
                min_votes=min_votes,
                min_purity=min_purity,
            )
            aggregates = aggregate_history_features(
                history,
                cutoff_year=cutoff_year,
                role=role,
                recent_windows=recent_windows,
                include_type_specific=include_type_specific,
            )
            features = _attach_aggregates(mapping, aggregates, role=role)
            mappings.append(mapping)
            if role == "pitcher":
                pitcher_outputs.append(features)
            else:
                batter_outputs.append(features)
            role_info[role] = _mapping_diagnostics(
                mapping, main, cutoff_year=cutoff_year, role=role
            )
        cutoff_info["roles"] = role_info
        cutoff_diagnostics[str(cutoff_year)] = cutoff_info

    diagnostics = {
        "leakage_policy": {
            "mapping_filter": "main/history alignment evidence season < cutoff_year",
            "aggregate_filter": "TrackMan season < cutoff_year",
            "current_pitch_measurements_emitted": False,
            "test_row_aggregation": False,
        },
        "alignment": alignment_diagnostics,
        "cutoffs": cutoff_diagnostics,
    }
    return HistoryFeatureBundle(
        pitcher_features=pd.concat(pitcher_outputs, ignore_index=True),
        batter_features=pd.concat(batter_outputs, ignore_index=True),
        entity_mapping=pd.concat(mappings, ignore_index=True),
        mapping_evidence=evidence,
        game_matches=matches,
        diagnostics=diagnostics,
    )

def join_selected_pitcher_history(
    rows: pd.DataFrame,
    selected_columns: Sequence[str],
    history_table: pd.DataFrame,
) -> pd.DataFrame:
    required = {"cutoff_year", "pitcher_id", *selected_columns}
    missing = required.difference(history_table.columns)
    if missing:
        raise ValueError(f"history lookup 열 누락: {sorted(missing)}")
    if history_table.duplicated(["cutoff_year", "pitcher_id"]).any():
        raise ValueError("history lookup key가 중복되어 있다")

    keys = pd.DataFrame({
        "__row_order": np.arange(len(rows), dtype=np.int64),
        "cutoff_year": pd.to_numeric(rows["season"], errors="raise").to_numpy(np.int32),
        "pitcher_id": pd.to_numeric(rows["pitcher_id"], errors="raise").to_numpy(np.int32),
    })
    subset = history_table[["cutoff_year", "pitcher_id", *selected_columns]]
    merged = keys.merge(
        subset,
        on=["cutoff_year", "pitcher_id"],
        how="left",
        sort=False,
        validate="many_to_one",
    ).sort_values("__row_order", kind="stable")
    output = merged[list(selected_columns)].copy()
    output.index = rows.index
    output["tm_pitcher_history_available"] = output.notna().any(axis=1).astype(np.int32)
    for column in selected_columns:
        output[column] = pd.to_numeric(output[column], errors="coerce").astype(np.float32)
    return output


def empty_history_lookup(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=["cutoff_year", "pitcher_id", *columns]).astype({
        "cutoff_year": np.int32,
        "pitcher_id": np.int32,
    })

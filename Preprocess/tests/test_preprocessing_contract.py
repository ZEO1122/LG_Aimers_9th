from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from Preprocess.common import (
    PipelineConfig,
    PreprocessingPipeline,
    build_mechanics_lookup,
)


def _rows() -> pd.DataFrame:
    size = 8
    seasons = np.array([2022, 2022, 2023, 2023, 2024, 2024, 2025, 2025])
    frame = pd.DataFrame(
        {
            "row_id": [f"ROW_{index:02d}" for index in range(size)],
            "season": seasons,
            "game_month": [3, 4, 5, 6, 7, 8, 9, 10],
            "game_dayofweek": [1, 2, 3, 4, 5, 6, 0, 1],
            "inning": [1, 2, 3, 4, 5, 6, 7, 8],
            "top_bottom": ["T", "B"] * 4,
            "game_type": ["R", "R", "F", "R", "F", "R", "F", "R"],
            "balls_before": [0, 1, 2, 3, 0, 1, 2, 3],
            "strikes_before": [0, 1, 2, 2, 1, 0, 2, 1],
            "outs_before": [0, 1, 2, 0, 1, 2, 0, 1],
            "run_top_before": [0, 0, 1, 1, 2, 2, 3, 3],
            "run_bot_before": [0, 1, 1, 2, 2, 3, 3, 4],
            "run_total_before": [0, 1, 2, 3, 4, 5, 6, 7],
            "score_diff_home": [0, 1, 0, -1, 2, -2, 1, 0],
            "score_diff_pitcher_team": [0, -1, 0, 1, -2, 2, -1, 0],
            "runner_on_1b": [0, 1, 0, 1, 0, 1, 0, 1],
            "runner_on_2b": [0, 0, 1, 1, 0, 0, 1, 1],
            "runner_on_3b": [0, 0, 0, 1, 1, 0, 0, 1],
            "num_runners_on": [0, 1, 1, 3, 1, 1, 1, 3],
            "base_state": ["___", "1__", "_2_", "123", "__3", "1__", "_2_", "123"],
            "home_win_expectancy": [50.0, 52.0, 48.0, 55.0, 45.0, 60.0, 40.0, 50.0],
            "away_win_expectancy": [50.0, 48.0, 52.0, 45.0, 55.0, 40.0, 60.0, 50.0],
            "li": [0.8, 1.0, 1.2, 1.5, 0.7, 2.0, 1.1, 0.9],
            "pitcher_id": [10, 10, 10, 11, 10, 11, 10, 11],
            "batter_id": [20, 21, 20, 21, 20, 21, 20, 21],
            "pitcher_hand": [1, 1, 1, 2, 1, 2, 1, 2],
            "batter_hand": [1, 2, 2, 1, 1, 2, 2, 1],
            "pitcher_team_id": [100, 100, 100, 101, 100, 101, 100, 101],
            "batter_team_id": [200, 201, 200, 201, 200, 201, 200, 201],
            "asof_pitcher_n": [0, 1, 5, 3, 9, 7, 12, 11],
            "asof_pitcher_success_rate": [np.nan, 1.0, 0.6, 1 / 3, 5 / 9, 4 / 7, 7 / 12, 6 / 11],
            "asof_pitcher_reverse_rate": [np.nan, 0.0, 0.2, 1 / 3, 2 / 9, 1 / 7, 3 / 12, 2 / 11],
            "asof_pitcher_middle_rate": [np.nan, 0.0, 0.2, 0.0, 1 / 9, 1 / 7, 2 / 12, 2 / 11],
            "asof_pitcher_ball_rate": [np.nan, 0.0, 0.4, 1 / 3, 3 / 9, 3 / 7, 4 / 12, 4 / 11],
            "asof_pitcher_strike_rate": [np.nan, 1.0, 0.4, 1 / 3, 4 / 9, 3 / 7, 5 / 12, 5 / 11],
            "asof_pitcher_prev1_game_success_rate": [np.nan, 1.0, 0.5, 0.4, 0.6, 0.55, 0.58, 0.52],
            "asof_pitcher_prev3_game_success_rate": [np.nan, 1.0, 0.55, 0.45, 0.58, 0.52, 0.56, 0.51],
            "asof_pitcher_prev5_game_success_rate": [np.nan, 1.0, 0.52, 0.44, 0.57, 0.51, 0.55, 0.50],
            "asof_pitcher_prev1_game_middle_rate": [np.nan, 0.0, 0.2, 0.1, 0.15, 0.12, 0.14, 0.13],
            "asof_pitcher_prev3_game_middle_rate": [np.nan, 0.0, 0.18, 0.12, 0.14, 0.13, 0.14, 0.13],
            "asof_pitcher_prev5_game_middle_rate": [np.nan, 0.0, 0.17, 0.11, 0.14, 0.13, 0.14, 0.13],
            "asof_batter_n": [0, 0, 2, 2, 4, 4, 6, 6],
            "asof_batter_success_rate": [np.nan, np.nan, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            "asof_batter_middle_rate": [np.nan, np.nan, 0.0, 0.5, 0.25, 0.25, 1 / 3, 1 / 3],
            "asof_pitcher_pitchmix_n": [0, 1, 5, 3, 9, 7, 12, 11],
            "asof_pitcher_fastball_rate": [np.nan, 1.0, 0.6, 2 / 3, 5 / 9, 4 / 7, 7 / 12, 6 / 11],
            "asof_pitcher_breaking_rate": [np.nan, 0.0, 0.2, 1 / 3, 2 / 9, 2 / 7, 3 / 12, 3 / 11],
            "asof_pitcher_offspeed_rate": [np.nan, 0.0, 0.2, 0.0, 2 / 9, 1 / 7, 2 / 12, 2 / 11],
            "control_success": [1, 0, 1, 0, 1, 0, 1, 0],
        }
    )
    return frame


class PreprocessingContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = _rows()
        cls.pipeline = PreprocessingPipeline(PipelineConfig(use_trackman=False)).fit(cls.rows)

    def test_batch_and_singleton_match(self) -> None:
        evaluation = self.rows.iloc[-2:].drop(columns="control_success")
        batch = self.pipeline.transform(evaluation)
        for offset, index in enumerate(evaluation.index):
            singleton = self.pipeline.transform(evaluation.loc[[index]])
            for key in ("base", "history_union", "semantic", "deep_frame", "intent_frame", "state_features"):
                assert_frame_equal(
                    batch[key].iloc[[offset]].reset_index(drop=True),
                    singleton[key].reset_index(drop=True),
                    check_dtype=True,
                )
            np.testing.assert_array_equal(
                batch["deep_tensor"]["numeric"][offset : offset + 1],
                singleton["deep_tensor"]["numeric"],
            )
            np.testing.assert_array_equal(
                batch["deep_tensor"]["categorical"][offset : offset + 1],
                singleton["deep_tensor"]["categorical"],
            )

    def test_row_permutation_does_not_change_features(self) -> None:
        evaluation = self.rows.iloc[-4:].drop(columns="control_success")
        direct = self.pipeline.transform(evaluation)
        shuffled_rows = evaluation.sample(frac=1.0, random_state=42)
        shuffled = self.pipeline.transform(shuffled_rows)
        for key in ("base", "semantic", "deep_frame", "intent_frame", "state_features"):
            left = direct[key].sort_index()
            right = shuffled[key].sort_index()
            assert_frame_equal(left, right, check_dtype=True)

    def test_transform_ignores_target_column(self) -> None:
        with_target = self.pipeline.transform(self.rows.iloc[-2:])
        without_target = self.pipeline.transform(self.rows.iloc[-2:].drop(columns="control_success"))
        assert_frame_equal(with_target["base"], without_target["base"])

    def test_mechanics_lookup_ignores_current_and_future_seasons(self) -> None:
        history = pd.DataFrame(
            {
                "season": [2023] * 4 + [2024] * 4,
                "trackman_game_id": [1] * 4 + [2] * 4,
                "pitch_no": [1, 2, 3, 4, 1, 2, 3, 4],
                "pitcher_trackman_id": [900] * 8,
                "pitcher_hand": ["Right"] * 8,
                "rel_side": [1.0, 1.1, 0.9, 1.2, 5.0, 5.1, 4.9, 5.2],
                "horz_break": [8.0, 7.5, -4.0, -4.5, 40.0, 41.0, -30.0, -31.0],
                "rel_height": [5.8, 5.9, 5.7, 6.0, 8.0, 8.1, 7.9, 8.2],
                "induced_vert_break": [16.0, 15.5, 4.0, 3.5, 30.0, 31.0, 20.0, 21.0],
                "rel_speed": [94.0, 93.0, 82.0, 81.0, 101.0, 100.0, 92.0, 91.0],
                "pitch_type_group": ["fastball", "fastball", "breaking", "breaking"] * 2,
            }
        )
        mapping = pd.DataFrame(
            {
                "cutoff_year": [2024],
                "role": ["pitcher"],
                "main_entity_id": [10],
                "trackman_entity_id": [900],
                "accepted": [True],
            }
        )
        original = build_mechanics_lookup(history, mapping, [2024])
        changed = history.copy()
        future = changed["season"].ge(2024)
        changed.loc[future, ["rel_side", "horz_break", "rel_height", "induced_vert_break", "rel_speed"]] = 999.0
        rebuilt = build_mechanics_lookup(changed, mapping, [2024])
        assert_frame_equal(original, rebuilt)


if __name__ == "__main__":
    unittest.main()

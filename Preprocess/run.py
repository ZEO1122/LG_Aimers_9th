"""Executable smoke check for the shared preprocessing package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .common import PipelineConfig, PreprocessingPipeline
else:
    from common import PipelineConfig, PreprocessingPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke", help="validate official CSV schema and row independence")
    smoke.add_argument("--data-dir", type=Path, required=True)
    smoke.add_argument("--train-rows", type=int, default=5_000)
    smoke.add_argument("--test-rows", type=int, default=5)
    return parser


def smoke(data_dir: Path, train_rows: int, test_rows: int) -> dict[str, object]:
    if train_rows < 1 or test_rows < 1:
        raise ValueError("train-rows and test-rows must be positive")
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(f"train.csv and test.csv are required under {data_dir}")

    train = pd.read_csv(train_path, nrows=train_rows, low_memory=False)
    test = pd.read_csv(test_path, nrows=test_rows, low_memory=False)
    if test.empty:
        raise ValueError("test.csv did not provide any rows")
    pipeline = PreprocessingPipeline(PipelineConfig(use_trackman=False)).fit(train)
    batch = pipeline.transform(test)
    singleton = pipeline.transform(test.iloc[[0]])
    numeric_equal = np.array_equal(
        batch["deep_tensor"]["numeric"][:1], singleton["deep_tensor"]["numeric"]
    )
    categorical_equal = np.array_equal(
        batch["deep_tensor"]["categorical"][:1],
        singleton["deep_tensor"]["categorical"],
    )
    if not numeric_equal or not categorical_equal:
        raise AssertionError("batch and singleton preprocessing differ")
    return {
        "status": "pass",
        "trackman_branch": "not_run",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "base_shape": list(batch["base"].shape),
        "deep_numeric_shape": list(batch["deep_tensor"]["numeric"].shape),
        "deep_categorical_shape": list(batch["deep_tensor"]["categorical"].shape),
        "row_independence": True,
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "smoke":
        result = smoke(args.data_dir.resolve(), args.train_rows, args.test_rows)
    else:
        raise AssertionError(f"unhandled command: {args.command}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

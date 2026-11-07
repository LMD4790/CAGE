#!/usr/bin/env python3
"""Prepare LightGBM class probabilities for training, validation, and test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import joblib

from cage.support import build_class_support
from cage.data import read_jsonl, validate_splits
from cage.provenance import create_manifest


def write_jsonl(path: Path, records: list[dict], support: np.ndarray) -> None:
    if len(records) != len(support):
        raise ValueError("Record and support counts differ")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record, vector in zip(records, support):
            output = dict(record)
            output.pop("ml_fusion_score", None)
            output["class_support"] = [float(value) for value in vector]
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, required=True)
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output-dir must be empty; existing artifacts are not overwritten")

    train = read_jsonl(args.train_jsonl)
    val = read_jsonl(args.val_jsonl)
    test = read_jsonl(args.test_jsonl)
    validate_splits({"train": train, "val": val, "test": test})
    for rows in (train, val, test):
        if any(not 0 <= row["label"] < args.num_classes for row in rows):
            parser.error("labels must be in [0, num-classes)")
    train_support, val_support, test_support, artifact = build_class_support(
        train,
        val,
        test,
        num_classes=args.num_classes,
        seed=args.seed,
        n_splits=args.folds,
        n_jobs=args.n_jobs,
    )
    write_jsonl(args.output_dir / "train.jsonl", train, train_support)
    write_jsonl(args.output_dir / "val.jsonl", val, val_support)
    write_jsonl(args.output_dir / "test.jsonl", test, test_support)
    joblib.dump(artifact, args.output_dir / "support_model.joblib")
    manifest = create_manifest(
        args.output_dir,
        {"train": args.train_jsonl, "val": args.val_jsonl, "test": args.test_jsonl},
        {"train": train, "val": val, "test": test},
        artifact,
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Wrote prepared splits to {args.output_dir}")


if __name__ == "__main__":
    main()

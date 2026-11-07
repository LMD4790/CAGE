#!/usr/bin/env python3
"""Predict new code pairs using a trained CAGE checkpoint."""

import argparse
import json
from pathlib import Path

import joblib
import torch
from torch.utils.data import DataLoader

from cage.data import PairDataset, read_jsonl
from cage.runtime import load_model, predict_loader
from cage.support import predict_class_support
from cage.provenance import verify_artifact, verify_precomputed_input


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--support-model", type=Path)
    parser.add_argument("--support-manifest", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.output_jsonl.exists():
        parser.error("output-jsonl already exists")
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    records = read_jsonl(args.input_jsonl, require_labels=False)
    model, tokenizer, metadata = load_model(args.model_dir, args.device)
    provenance = metadata.get("support_provenance", {})
    if args.support_model:
        verify_artifact(args.support_model, provenance.get("support_artifact_sha256"))
        # joblib artifacts must come from a trusted source.
        artifact = joblib.load(args.support_model)
        if artifact["num_classes"] != model.num_classes:
            parser.error("support artifact and CAGE checkpoint use different class counts")
        probabilities = predict_class_support(records, artifact)
        for row, vector in zip(records, probabilities):
            row["class_support"] = vector.tolist()
    else:
        verify_precomputed_input(
            args.input_jsonl, len(records), model.num_classes, provenance, args.support_manifest
        )
    data = PairDataset(
        records, tokenizer, metadata["max_length"], metadata["language"], model.num_classes,
        include_labels=False,
    )
    _, probabilities = predict_loader(
        model, DataLoader(data, batch_size=args.batch_size), args.device
    )
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for index, vector in enumerate(probabilities):
            handle.write(json.dumps({
                "index": index,
                "id": records[index].get("id", records[index].get("pair_id", index)),
                "prediction": int(vector.argmax()), "probabilities": vector.tolist(),
            }) + "\n")
    print(f"Wrote {len(records)} predictions to {args.output_jsonl}")


if __name__ == "__main__":
    main()

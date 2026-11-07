#!/usr/bin/env python3
"""Fine-tune CAGE on explicit training, validation, and test partitions."""

import argparse
import importlib.metadata
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import RobertaModel, RobertaTokenizer, get_linear_schedule_with_warmup

from cage import CAGE
from cage.data import PairDataset, file_sha256, read_jsonl, support_vector, validate_splits
from cage.runtime import METRIC_NAMES, load_model, metrics, predict_loader, save_model
from cage.provenance import verify_training_provenance


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def optimizer_groups(model, encoder_lr, fusion_lr):
    # The AST value encoder shares the lower encoder learning rate.
    low_rate = list(model.encoder.parameters()) + list(model.ast_value_encoder.parameters())
    low_ids = {id(parameter) for parameter in low_rate}
    high_rate = [parameter for parameter in model.parameters() if id(parameter) not in low_ids]
    return [
        {"params": low_rate, "lr": encoder_lr},
        {"params": high_rate, "lr": fusion_lr},
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ("train", "val", "test"):
        parser.add_argument(f"--{split}-jsonl", type=Path, required=True)
    parser.add_argument("--model-name", default="microsoft/codebert-base")
    parser.add_argument("--support-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--language", choices=["java", "c"], default="java")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--fusion-lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not 4 <= args.max_length <= 512:
        parser.error("epochs/batch-size must be positive; max-length must be in [4, 512]")
    if args.warmup_steps < 0 or args.encoder_lr <= 0 or args.fusion_lr <= 0:
        parser.error("learning rates must be positive; warmup-steps cannot be negative")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output-dir must be empty; existing results are not overwritten")
    if args.fp16 and not str(args.device).startswith("cuda"):
        parser.error("--fp16 requires a CUDA device")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    split_paths = {name: getattr(args, name + "_jsonl") for name in ("train", "val", "test")}
    splits = {name: read_jsonl(path) for name, path in split_paths.items()}
    validate_splits(splits)
    row = splits["train"][0]
    num_classes = len(row.get("class_support", row.get("ml_fusion_score", [])))
    if num_classes < 2:
        parser.error("prepared inputs require class_support vectors with at least two classes")
    for rows in splits.values():
        for row in rows:
            support_vector(row, num_classes)
            if not 0 <= row["label"] < num_classes:
                parser.error("labels must be in [0, number of classes)")
    if set(row["label"] for row in splits["train"]) != set(range(num_classes)):
        parser.error("training split must contain every declared class")
    provenance = verify_training_provenance(split_paths, splits, num_classes, args.support_manifest)

    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    encoder = RobertaModel.from_pretrained(args.model_name)
    max_positions = encoder.config.max_position_embeddings - encoder.config.pad_token_id - 1
    if args.max_length > max_positions:
        parser.error(f"encoder supports at most {max_positions} nonpadding tokens")
    model = CAGE(encoder, num_classes).to(args.device)
    datasets = {
        name: PairDataset(rows, tokenizer, args.max_length, args.language, num_classes)
        for name, rows in splits.items()
    }
    loaders = {
        name: DataLoader(data, batch_size=args.batch_size, shuffle=name == "train")
        for name, data in datasets.items()
    }
    optimizer = torch.optim.AdamW(
        optimizer_groups(model, args.encoder_lr, args.fusion_lr), eps=1e-8, weight_decay=0.0
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, args.warmup_steps, len(loaders["train"]) * args.epochs
    )
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=args.fp16)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "num_classes": num_classes,
        "support_provenance": provenance,
        "splits": {
            name: {"count": len(splits[name]), "sha256": file_sha256(path)}
            for name, path in split_paths.items()
        },
        "versions": {
            package: importlib.metadata.version(package)
            for package in ("torch", "transformers", "numpy", "lightgbm", "tree-sitter", "tree-sitter-languages")
        },
        "selection_metric": "validation_macro_f1",
        "test_policy": "one evaluation after validation checkpoint selection",
        "optimizer_weight_decay": 0.0,
    }
    write_json(args.output_dir / "run_config.json", run_config)
    history, best_f1, best_epoch = [], -1.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, count = 0.0, 0
        for batch in loaders["train"]:
            labels = batch.pop("label").to(args.device)
            inputs = {key: value.to(args.device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=args.fp16):
                loss = criterion(model(**inputs), labels)
            scaler.scale(loss).backward()
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= previous_scale:
                scheduler.step()
            total_loss += float(loss.detach()) * len(labels)
            count += len(labels)
        actual, probabilities = predict_loader(model, loaders["val"], args.device)
        validation = metrics(actual, probabilities.argmax(1), num_classes)
        record = {
            "epoch": epoch, "train_loss": total_loss / count,
            "validation": {key: validation[key] for key in METRIC_NAMES},
        }
        history.append(record)
        write_json(args.output_dir / "history.json", history)
        if validation["macro_f1"] > best_f1:
            best_f1, best_epoch = validation["macro_f1"], epoch
            save_model(model, tokenizer, args.output_dir, {
                "best_epoch": epoch, "validation_macro_f1": best_f1,
                "max_length": args.max_length, "language": args.language,
                "support_provenance": provenance,
            })
        print(json.dumps(record), flush=True)

    del model, optimizer, scheduler, scaler, encoder, inputs, loss
    if str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()
    model, _, _ = load_model(args.output_dir, args.device)
    (args.output_dir / "test_started.json").write_text(
        json.dumps({"best_epoch": best_epoch, "test_sha256": file_sha256(args.test_jsonl)}),
        encoding="utf-8",
    )
    actual, probabilities = predict_loader(model, loaders["test"], args.device)
    predicted = probabilities.argmax(1)
    test = metrics(actual, predicted, num_classes)
    write_json(args.output_dir / "test_metrics.json", {
        "best_epoch": best_epoch, "best_validation_macro_f1": best_f1, "test": test
    })
    with (args.output_dir / "test_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for index, row in enumerate(splits["test"]):
            handle.write(json.dumps({
                "index": index, "id": row.get("id", row.get("pair_id", index)),
                "label": actual[index], "prediction": int(predicted[index]),
                "probabilities": probabilities[index].tolist(),
            }) + "\n")
    print(json.dumps({"best_epoch": best_epoch, "test": {key: test[key] for key in METRIC_NAMES}}))


if __name__ == "__main__":
    main()

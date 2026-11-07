"""Model serialization, prediction, and classification metrics."""

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score, recall_score,
)
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

from .model import CAGE


METRIC_NAMES = ("macro_f1", "weighted_f1", "accuracy", "macro_recall", "weighted_recall")


def metrics(labels, predictions, num_classes):
    classes = list(range(num_classes))
    return {
        "macro_f1": float(f1_score(labels, predictions, labels=classes, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, labels=classes, average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_recall": float(recall_score(labels, predictions, labels=classes, average="macro", zero_division=0)),
        "weighted_recall": float(recall_score(labels, predictions, labels=classes, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=classes).tolist(),
        "per_class": classification_report(labels, predictions, labels=classes, output_dict=True, zero_division=0),
        "sample_count": len(labels),
        "units": "fraction",
    }


def predict_loader(model, loader, device):
    model.eval()
    actual, probabilities = [], []
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("label")
            inputs = {key: value.to(device) for key, value in batch.items()}
            logits = model(**inputs)
            probabilities.extend(torch.softmax(logits, dim=-1).cpu().tolist())
            actual.extend(labels.tolist())
    return actual, np.asarray(probabilities, dtype=np.float32)


def save_model(model, tokenizer, output_dir, metadata):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.encoder.config.save_pretrained(output_dir / "encoder_config")
    tokenizer.save_pretrained(output_dir / "tokenizer")
    checkpoint = {
        "format_version": 1,
        "model_state_dict": {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        },
        "num_classes": model.num_classes,
        "model_config": {
            "num_ast_features": model.num_ast_features,
            "num_attention_heads": model.cross_attention.num_heads,
        },
        "metadata": metadata,
    }
    temporary = output_dir / "best_model.pt.tmp"
    torch.save(checkpoint, temporary)
    temporary.replace(output_dir / "best_model.pt")


def load_model(output_dir, device="cpu"):
    output_dir = Path(output_dir)
    checkpoint = torch.load(
        output_dir / "best_model.pt", map_location="cpu", weights_only=True
    )
    if checkpoint["format_version"] != 1:
        raise ValueError("Unsupported checkpoint format")
    config = RobertaConfig.from_pretrained(output_dir / "encoder_config", local_files_only=True)
    tokenizer = RobertaTokenizer.from_pretrained(output_dir / "tokenizer", local_files_only=True)
    model_config = checkpoint.get("model_config", {})
    model = CAGE(
        RobertaModel(config), checkpoint["num_classes"],
        num_ast_features=model_config.get("num_ast_features", 18),
        num_attention_heads=model_config.get("num_attention_heads", 8),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, tokenizer, checkpoint["metadata"]

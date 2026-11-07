"""Out-of-fold LightGBM probabilities for code-pair classification."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold

from .features import COMPARISON_FEATURE_NAMES, pairwise_comparison_features, tokenize_code
from .data import IDENTITY_POLICY, OOF_POLICY, training_groups, validate_splits


def _codes(records: Sequence[dict]) -> list[str]:
    return [str(record.get("code1", "") or "") for record in records] + [
        str(record.get("code2", "") or "") for record in records
    ]


def _fit_vectorizer(records: Sequence[dict]) -> TfidfVectorizer:
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize_code,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        dtype=np.float64,
    )
    codes = list(dict.fromkeys(_codes(records)))
    if not any(tokenize_code(code) for code in codes):
        raise ValueError("TF-IDF training corpus contains no code tokens")
    vectorizer.fit(codes)
    return vectorizer


def _feature_matrix(records: Sequence[dict], vectorizer: TfidfVectorizer) -> np.ndarray:
    if not records:
        return np.empty((0, 20), dtype=np.float64)
    encoded = vectorizer.transform(_codes(records))
    count = len(records)
    tfidf_scores = np.asarray(
        encoded[:count].multiply(encoded[count:]).sum(axis=1)
    ).ravel()
    return np.stack(
        [
            pairwise_comparison_features(
                str(record.get("code1", "") or ""),
                str(record.get("code2", "") or ""),
                float(tfidf_scores[index]),
            )
            for index, record in enumerate(records)
        ]
    )


def _lightgbm(num_classes: int, seed: int, n_jobs: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="multiclass",
        num_class=num_classes,
        n_estimators=100,
        num_leaves=166,
        max_depth=-1,
        learning_rate=0.040356537,
        colsample_bytree=0.6010569422,
        reg_alpha=0.0005011127,
        reg_lambda=0.7646112918,
        class_weight=None,
        random_state=seed,
        n_jobs=n_jobs,
        verbosity=-1,
    )


def _aligned_probabilities(model, features: np.ndarray, num_classes: int) -> np.ndarray:
    if len(features) == 0:
        return np.empty((0, num_classes), dtype=np.float32)
    raw = model.booster_.predict(features)
    aligned = np.zeros((len(features), num_classes), dtype=np.float32)
    aligned[:, np.asarray(model.classes_, dtype=np.int64)] = raw
    return aligned


def build_class_support(
    train_records: Sequence[dict],
    val_records: Sequence[dict],
    test_records: Sequence[dict],
    num_classes: int,
    seed: int = 2028,
    n_splits: int = 5,
    n_jobs: int = -1,
):
    """Return OOF training and full-training-model validation/test vectors."""
    if not train_records:
        raise ValueError("training split is empty")
    if n_splits < 2 or num_classes < 2:
        raise ValueError("At least two classes and two OOF folds are required")
    validate_splits({"train": train_records, "val": val_records, "test": test_records})
    if any(type(row.get("label")) is not int for row in train_records):
        raise ValueError("training labels must be integers")
    labels = np.asarray([row["label"] for row in train_records], dtype=np.int64)
    if labels.min() < 0 or labels.max() >= num_classes:
        raise ValueError("training labels must be in [0, num_classes)")
    counts = np.bincount(labels, minlength=num_classes)
    if np.any(counts < n_splits):
        raise ValueError(
            f"each class needs at least {n_splits} training examples for OOF support"
        )

    oof = np.zeros((len(train_records), num_classes), dtype=np.float32)
    groups = training_groups(train_records)
    if any(len(np.unique(groups[labels == label])) < n_splits for label in range(num_classes)):
        raise ValueError(f"Each class needs at least {n_splits} independent source/group components")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = splitter.split(np.zeros(len(labels)), labels, groups)
    assignments = np.full(len(labels), -1, dtype=np.int64)
    for fold, (fit_indices, heldout_indices) in enumerate(folds):
        assignments[heldout_indices] = fold
        if len(np.unique(labels[fit_indices])) != num_classes:
            raise ValueError("An OOF training fold is missing a class; revise group/fold settings")
        fit_records = [train_records[index] for index in fit_indices]
        heldout_records = [train_records[index] for index in heldout_indices]
        vectorizer = _fit_vectorizer(fit_records)
        fit_features = _feature_matrix(fit_records, vectorizer)
        heldout_features = _feature_matrix(heldout_records, vectorizer)
        model = _lightgbm(num_classes, seed, n_jobs)
        model.fit(fit_features, labels[fit_indices])
        oof[heldout_indices] = _aligned_probabilities(
            model, heldout_features, num_classes
        )

    final_vectorizer = _fit_vectorizer(train_records)
    train_features = _feature_matrix(train_records, final_vectorizer)
    final_model = _lightgbm(num_classes, seed, n_jobs)
    final_model.fit(train_features, labels)
    val_features = _feature_matrix(val_records, final_vectorizer)
    test_features = _feature_matrix(test_records, final_vectorizer)
    val_support = _aligned_probabilities(final_model, val_features, num_classes)
    test_support = _aligned_probabilities(final_model, test_features, num_classes)
    artifact = {
        "schema_version": 2,
        "feature_names": list(COMPARISON_FEATURE_NAMES),
        "vectorizer": final_vectorizer,
        "model": final_model,
        "num_classes": num_classes,
        "seed": seed,
        "folds": n_splits,
        "grouped_oof": True,
        "identity_policy": IDENTITY_POLICY,
        "oof_policy": OOF_POLICY,
        "oof_assignments": assignments.tolist(),
        "training_groups": groups.tolist(),
        "label_ids": list(range(num_classes)),
    }
    return oof, val_support, test_support, artifact


def predict_class_support(records: Sequence[dict], artifact: dict) -> np.ndarray:
    """Generate support for new pairs using a training-only fitted artifact."""
    version = artifact.get("schema_version")
    if version not in (1, 2):
        raise ValueError("Unsupported class-support artifact version")
    if version == 2 and artifact.get("feature_names") != list(COMPARISON_FEATURE_NAMES):
        raise ValueError("Class-support artifact feature order does not match this release")
    features = _feature_matrix(records, artifact["vectorizer"])
    if version == 1:
        # Schema 1 places ratio/difference before summary statistics.
        features = features[:, [*range(13), 17, 18, 13, 14, 15, 16, 19]]
    return _aligned_probabilities(
        artifact["model"],
        features,
        artifact["num_classes"],
    )

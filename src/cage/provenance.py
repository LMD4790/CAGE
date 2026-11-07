"""Verify prepared data, fold assignments, and support-model checksums."""

import json
from pathlib import Path

from .data import IDENTITY_POLICY, OOF_POLICY, file_sha256, training_groups
from .features import COMPARISON_FEATURE_NAMES


def create_manifest(output_dir, split_paths, splits, artifact):
    output_dir = Path(output_dir)
    return {
        "schema_version": 2,
        "support_artifact_schema_version": artifact["schema_version"],
        "feature_names": list(COMPARISON_FEATURE_NAMES),
        "identity_policy": IDENTITY_POLICY,
        "oof_policy": OOF_POLICY,
        "seed": artifact["seed"],
        "folds": artifact["folds"],
        "num_classes": artifact["num_classes"],
        "label_ids": list(range(artifact["num_classes"])),
        "train_support": "out_of_fold",
        "heldout_support": "fit_on_training_only",
        "oof_assignments": artifact["oof_assignments"],
        "support_artifact_sha256": file_sha256(output_dir / "support_model.joblib"),
        "splits": {
            name: {
                "count": len(rows),
                "input_sha256": file_sha256(split_paths[name]),
                "prepared_sha256": file_sha256(output_dir / f"{name}.jsonl"),
            }
            for name, rows in splits.items()
        },
    }


def manifest_path_for(paths, explicit=None):
    if explicit is not None:
        return Path(explicit)
    parents = {Path(path).resolve().parent for path in paths}
    if len(parents) != 1:
        raise ValueError("Prepared splits in different folders require --support-manifest")
    return parents.pop() / "manifest.json"


def load_manifest(path, num_classes):
    path = Path(path)
    if not path.is_file():
        raise ValueError("Missing support manifest; generate prepared splits with scripts/prepare_support.py")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 2,
        "support_artifact_schema_version": 2,
        "feature_names": list(COMPARISON_FEATURE_NAMES),
        "identity_policy": IDENTITY_POLICY,
        "oof_policy": OOF_POLICY,
        "num_classes": num_classes,
        "label_ids": list(range(num_classes)),
        "train_support": "out_of_fold",
        "heldout_support": "fit_on_training_only",
    }
    if not isinstance(manifest, dict) or any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError("Support manifest schema, class ordering or preparation protocol mismatch")
    digest = manifest.get("support_artifact_sha256", "")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Support manifest has no valid artifact SHA-256")
    entries = manifest.get("splits")
    if not isinstance(entries, dict) or set(entries) != {"train", "val", "test"}:
        raise ValueError("Support manifest must declare train, val and test splits")
    for entry in entries.values():
        if not isinstance(entry, dict) or type(entry.get("count")) is not int or entry["count"] < 1:
            raise ValueError("Invalid prepared split entry in support manifest")
    return manifest


def verify_artifact(path, expected_sha256):
    if not expected_sha256:
        raise ValueError("Checkpoint is missing its support-model checksum; train with the current preparation files")
    if not Path(path).is_file() or file_sha256(path) != expected_sha256:
        raise ValueError("Support artifact SHA-256 differs from the bound artifact")


def verify_training_provenance(split_paths, splits, num_classes, manifest_path=None):
    path = manifest_path_for(split_paths.values(), manifest_path)
    manifest = load_manifest(path, num_classes)
    for name, file_path in split_paths.items():
        entry = manifest["splits"][name]
        if entry["count"] != len(splits[name]) or entry.get("prepared_sha256") != file_sha256(file_path):
            raise ValueError(f"Prepared {name} split differs from its support manifest")
    verify_artifact(path.parent / "support_model.joblib", manifest["support_artifact_sha256"])
    assignments = manifest.get("oof_assignments")
    folds = manifest.get("folds")
    if type(folds) is not int or folds < 2 or not isinstance(assignments, list):
        raise ValueError("Invalid OOF fold declaration")
    if len(assignments) != len(splits["train"]) or any(type(f) is not int or not 0 <= f < folds for f in assignments):
        raise ValueError("Invalid OOF fold assignments")
    if set(assignments) != set(range(folds)):
        raise ValueError("OOF assignments must cover every declared fold")
    groups = training_groups(splits["train"])
    seen = {}
    for group, fold in zip(groups, assignments):
        if int(group) in seen and seen[int(group)] != fold:
            raise ValueError("Shared source or declared group crosses OOF folds")
        seen[int(group)] = fold
    for heldout in range(folds):
        fit_labels = {row["label"] for row, fold in zip(splits["train"], assignments) if fold != heldout}
        if fit_labels != set(range(num_classes)):
            raise ValueError("An OOF training fold is missing a class")
    return {
        "manifest_schema_version": 2,
        "manifest_sha256": file_sha256(path),
        "support_artifact_sha256": manifest["support_artifact_sha256"],
        "identity_policy": IDENTITY_POLICY,
        "oof_policy": OOF_POLICY,
        "label_ids": list(range(num_classes)),
    }


def verify_precomputed_input(input_path, count, num_classes, provenance, manifest_path=None):
    path = manifest_path_for([input_path], manifest_path)
    manifest = load_manifest(path, num_classes)
    if not provenance or manifest["support_artifact_sha256"] != provenance.get("support_artifact_sha256"):
        raise ValueError("Input support provenance differs from the checkpoint")
    if file_sha256(path) != provenance.get("manifest_sha256"):
        raise ValueError("Input support manifest differs from the checkpoint")
    digest = file_sha256(input_path)
    if not any(entry.get("prepared_sha256") == digest and entry["count"] == count for entry in manifest["splits"].values()):
        raise ValueError("Precomputed input differs from its support manifest")

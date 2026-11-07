import json

import pytest

from cage.data import file_sha256
from cage.provenance import (
    create_manifest, load_manifest, verify_artifact,
    verify_precomputed_input, verify_training_provenance,
)


@pytest.fixture
def prepared(tmp_path):
    splits = {
        "train": [{"code1": f"train_{i}", "code2": f"other_{i}", "label": i % 2,
                   "class_support": [0.4, 0.6]} for i in range(8)],
        "val": [{"code1": "val_a", "code2": "val_b", "label": 0, "class_support": [0.4, 0.6]}],
        "test": [{"code1": "test_a", "code2": "test_b", "label": 1, "class_support": [0.4, 0.6]}],
    }
    paths = {name: tmp_path / f"{name}.jsonl" for name in splits}
    for name, path in paths.items():
        path.write_text(chr(10).join(json.dumps(row) for row in splits[name]), encoding="utf-8")
    (tmp_path / "support_model.joblib").write_bytes(b"opaque artifact for checksum tests")
    artifact = {"schema_version": 2, "seed": 2028, "folds": 2, "num_classes": 2,
                "oof_assignments": [0, 0, 1, 1, 0, 0, 1, 1]}
    manifest = create_manifest(tmp_path, paths, splits, artifact)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return paths, splits, manifest, manifest_path


def test_valid_provenance_binds_training_and_precomputed_prediction(prepared):
    paths, splits, _, path = prepared
    provenance = verify_training_provenance(paths, splits, 2)
    assert provenance["manifest_sha256"] == file_sha256(path)
    verify_precomputed_input(paths["test"], 1, 2, provenance)


@pytest.mark.parametrize("field,value", [
    ("class_support", [0.9, 0.1]), ("label", 1),
    ("semantic_code1", "transformed text"), ("ast_relations", [[0, 1]] * 18),
])
def test_modified_prepared_fields_are_rejected(prepared, field, value):
    paths, splits, _, _ = prepared
    splits["train"][0][field] = value
    paths["train"].write_text(chr(10).join(json.dumps(r) for r in splits["train"]), encoding="utf-8")
    with pytest.raises(ValueError, match="Prepared train split differs"):
        verify_training_provenance(paths, splits, 2)


@pytest.mark.parametrize("field,value", [
    ("schema_version", 1), ("label_ids", [1, 0]), ("feature_names", []),
    ("oof_policy", "row_folds"), ("num_classes", 3), ("train_support", "in_sample"),
])
def test_wrong_protocol_and_class_order_rejected(prepared, field, value):
    _, _, manifest, path = prepared
    manifest[field] = value
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol mismatch"):
        load_manifest(path, 2)


def test_missing_manifest_rejected(prepared):
    paths, splits, _, path = prepared
    path.unlink()
    with pytest.raises(ValueError, match="Missing support manifest"):
        verify_training_provenance(paths, splits, 2)


def test_wrong_artifact_rejected_before_deserialization(prepared):
    paths, splits, manifest, path = prepared
    artifact = path.parent / "support_model.joblib"
    artifact.write_bytes(b"replaced artifact")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_artifact(artifact, manifest["support_artifact_sha256"])
    with pytest.raises(ValueError, match="SHA-256"):
        verify_training_provenance(paths, splits, 2)


def test_crossing_group_assignments_rejected(prepared):
    paths, splits, manifest, path = prepared
    splits["train"][2]["code1"] = splits["train"][0]["code1"]
    paths["train"].write_text(chr(10).join(json.dumps(r) for r in splits["train"]), encoding="utf-8")
    manifest["splits"]["train"]["prepared_sha256"] = file_sha256(paths["train"])
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="crosses OOF folds"):
        verify_training_provenance(paths, splits, 2)


def test_unbound_checkpoint_and_changed_precomputed_input_rejected(prepared):
    paths, splits, _, _ = prepared
    with pytest.raises(ValueError, match="provenance differs"):
        verify_precomputed_input(paths["test"], 1, 2, {})
    provenance = verify_training_provenance(paths, splits, 2)
    paths["test"].write_text("modified", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from its support manifest"):
        verify_precomputed_input(paths["test"], 1, 2, provenance)


def test_changed_manifest_cannot_be_used_with_bound_checkpoint(prepared):
    paths, splits, manifest, path = prepared
    provenance = verify_training_provenance(paths, splits, 2)
    manifest["seed"] = 99
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest differs from the checkpoint"):
        verify_precomputed_input(paths["test"], 1, 2, provenance)

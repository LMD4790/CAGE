import joblib
import numpy as np
import pytest
from lightgbm import LGBMClassifier

from cage.support import _feature_matrix, _fit_vectorizer, build_class_support, predict_class_support
from cage.features import COMPARISON_FEATURE_NAMES


def rows():
    return [
        {"code1": f"int value_{i}(int x) {{ return x; }}",
         "code2": f"int other_{i}(int z) {{ return z + {i}; }}", "label": i % 2}
        for i in range(20)
    ]


def test_tfidf_matches_each_pair_without_dense_cross_pair_matrix():
    data = [
        {"code1": "alpha alpha beta", "code2": "alpha alpha beta"},
        {"code1": "unique_first", "code2": "unique_second"},
        {"code1": "rare word", "code2": "rare different"},
    ]
    vectorizer = _fit_vectorizer(data)
    matrix = _feature_matrix(data, vectorizer)
    assert np.isclose(matrix[0, 5], 1.0)
    assert np.isclose(matrix[1, 5], 0.0)
    for i, row in enumerate(data):
        a, b = vectorizer.transform([row["code1"], row["code2"]])
        assert np.isclose(matrix[i, 5], a.multiply(b).sum())


def test_support_ignores_heldout_labels_and_survives_serialization(tmp_path):
    train = rows()
    val = [dict(row, code1="val_" + row["code1"], code2="val_" + row["code2"]) for row in train[:2]]
    test = [dict(row, code1="test_" + row["code1"], code2="test_" + row["code2"]) for row in train[2:4]]
    a = build_class_support(train, val, test, 2, n_splits=2, n_jobs=1)
    changed = [dict(row, label=1-row["label"]) for row in val]
    changed_test = [dict(row, label=1-row["label"]) for row in test]
    b = build_class_support(train, changed, changed_test, 2, n_splits=2, n_jobs=1)
    for probabilities, again in zip(a[:3], b[:3]):
        assert np.allclose(probabilities, again)
        assert np.allclose(probabilities.sum(1), 1)
    path = tmp_path / "support.joblib"
    joblib.dump(a[3], path)
    restored = joblib.load(path)
    assert np.allclose(predict_class_support(test, restored), a[2])


def test_vectorizer_has_no_heldout_vocabulary():
    vectorizer = _fit_vectorizer([{"code1": "trainingword", "code2": "moretraining"}])
    _feature_matrix([{"code1": "heldoutword", "code2": "anotherheldout"}], vectorizer)
    assert "heldoutword" not in vectorizer.vocabulary_


def test_schema_one_support_preserves_prediction_after_column_reorder(tmp_path):
    data = [{'code1': 'int f() { ' + 'int v = 1; ' * i + 'return 1; }',
             'code2': 'int g() { return 1; }', 'label': int(i > 20)}
            for i in range(1, 41)]
    vectorizer = _fit_vectorizer(data)
    canonical = _feature_matrix(data, vectorizer)
    previous = canonical[:, [*range(13), 17, 18, 13, 14, 15, 16, 19]]
    labels = np.asarray([row['label'] for row in data])
    model = LGBMClassifier(objective='multiclass', num_class=2, n_estimators=10,
                          min_child_samples=1, min_data_in_bin=1, verbosity=-1, n_jobs=1)
    # Isolate the schema-1 ratio column so wrong ordering cannot pass by
    # relying on the first thirteen unchanged features.
    training = np.zeros_like(previous)
    training[:, 13] = previous[:, 13]
    model.fit(training, labels)
    expected = model.booster_.predict(previous)
    assert np.ptp(expected[:, 0]) > 0.01
    assert not np.allclose(model.booster_.predict(canonical), expected)
    artifact = {'schema_version': 1, 'num_classes': 2, 'model': model, 'vectorizer': vectorizer}
    path = tmp_path / 'schema_one_support.joblib'
    joblib.dump(artifact, path)
    np.testing.assert_allclose(predict_class_support(data, joblib.load(path)), expected, atol=1e-7)


def test_support_artifact_rejects_mismatched_named_schema():
    artifact = {'schema_version': 2, 'feature_names': list(reversed(COMPARISON_FEATURE_NAMES))}
    with pytest.raises(ValueError, match='feature order'):
        predict_class_support(rows(), artifact)


def test_shared_sources_never_cross_oof_folds():
    train = rows()
    for index in range(0, len(train), 2):
        train[index + 1]["code1"] = train[index]["code1"]
    artifact = build_class_support(train, [], [], 2, n_splits=5, n_jobs=1)[3]
    for index in range(0, len(train), 2):
        assert artifact["oof_assignments"][index] == artifact["oof_assignments"][index + 1]
    assert artifact["grouped_oof"] is True
    assert len(set(artifact["oof_assignments"])) == 5


def test_insufficient_independent_groups_cannot_fall_back_to_row_folds():
    train = rows()
    for row in train:
        row["code1"] = "int shared() { return 0; }"
    with pytest.raises(ValueError, match="independent source/group"):
        build_class_support(train, [], [], 2, n_splits=2, n_jobs=1)


def test_support_builder_rejects_overlapping_heldout_sources():
    train = rows()
    with pytest.raises(ValueError, match="Shared source"):
        build_class_support(train, [{"code1": train[0]["code1"], "code2": "different"}], [], 2, n_splits=2)


@pytest.mark.parametrize("label", [0.5, "1", True])
def test_support_builder_rejects_noninteger_labels(label):
    train = rows()
    train[0]["label"] = label
    with pytest.raises(ValueError, match="labels must be integers"):
        build_class_support(train, [], [], 2, n_splits=2, n_jobs=1)

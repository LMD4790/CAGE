import json

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from cage.data import PairDataset, code_identity, read_jsonl, support_vector, training_groups, validate_splits
from cage.features import extract_ast_features, pairwise_ast_relations, tokenize_code
from cage.runtime import metrics


def test_comment_removal_preserves_following_lines():
    assert "after_comment" in tokenize_code("// comment\nint after_comment = 1;")


@pytest.mark.parametrize("language,code", [
    ("java", "class A { int f(int x) { if(x>0) return x; return 0; } }"),
    ("c", "int f(int x) { if(x>0) return x; return 0; }"),
])
def test_ast_features_and_pair_symmetry(language, code):
    values = extract_ast_features(code, language)
    assert values.shape == (18,)
    assert values[0] > 1
    assert values[8] > 0
    identical = pairwise_ast_relations(code, code, language)
    assert np.allclose(identical[:, 0], 0)
    assert np.allclose(identical[:, 1], 1)
    other = code.replace("if(x>0)", "while(x>0)")
    assert np.allclose(
        pairwise_ast_relations(code, other, language),
        pairwise_ast_relations(other, code, language),
    )


def test_invalid_support_rejected():
    with pytest.raises(ValueError):
        support_vector({"class_support": [0.2, float("nan")]}, 2)
    with pytest.raises(ValueError):
        support_vector({"class_support": [0.9, 0.9]}, 2)


def test_reversed_pair_overlap_rejected():
    with pytest.raises(ValueError, match="Duplicate code pair"):
        validate_splits({
            "train": [{"code1": "a", "code2": "b"}],
            "test": [{"code1": "b", "code2": "a"}],
        })


def test_macro_metric_uses_fixed_label_universe():
    result = metrics([0, 0], [0, 0], 2)
    assert result["accuracy"] == 1
    assert result["macro_f1"] == 0.5


@pytest.mark.parametrize("same_split", [True, False])
def test_spacing_equivalent_pairs_are_rejected(same_split):
    first = {"code1": "int f(){return 1;}", "code2": "int g(){return 2;}"}
    second = {"code1": "int g ( ) { return 2 ; }", "code2": "int f ( ) { return 1 ; }"}
    splits = {"train": [first, second]} if same_split else {"train": [first], "test": [second]}
    with pytest.raises(ValueError, match="Duplicate code pair"):
        validate_splits(splits)


def test_conflicting_pair_labels_rejected():
    with pytest.raises(ValueError, match="conflicting labels"):
        validate_splits({"train": [{"code1": "a", "code2": "b", "label": 0},
                                   {"code1": "b", "code2": "a", "label": 1}]})


def test_single_shared_fragment_across_splits_rejected():
    with pytest.raises(ValueError, match="Shared source"):
        validate_splits({"train": [{"code1": "int f(){}", "code2": "a"}],
                         "test": [{"code1": "int f ( ) { }", "code2": "c"}]})


def test_identity_preserves_literals_comments_and_operator_boundaries():
    assert code_identity('return "a  b";') != code_identity('return "a b";')
    assert code_identity("return a + +b;") != code_identity("return a++ + b;")
    assert code_identity("a + + b") != code_identity("a++b")
    assert code_identity("// a b") != code_identity("// ab")
    assert code_identity('return "a \\" b";') == code_identity('return  "a \\" b" ;')


def test_transitive_source_and_declared_groups_join():
    rows = [{"code1": "a", "code2": "b", "group_id": "g1"},
            {"code1": "b", "code2": "c", "group_id": "g2"},
            {"code1": "d", "code2": "e", "group_id": "g2"},
            {"code1": "f", "code2": "g", "group_id": "g3"}]
    groups = training_groups(rows)
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]


@pytest.mark.parametrize("group", [None, "", [], True])
def test_invalid_declared_group_rejected(group):
    with pytest.raises(ValueError, match="group_id"):
        validate_splits({"train": [{"code1": "a", "code2": "b", "group_id": group}]})


def test_partial_or_cross_split_declared_groups_rejected():
    with pytest.raises(ValueError, match="every row"):
        validate_splits({"train": [{"code1": "a", "code2": "b", "group_id": "g"}],
                         "test": [{"code1": "c", "code2": "d"}]})
    with pytest.raises(ValueError, match="group_id occurs"):
        validate_splits({"train": [{"code1": "a", "code2": "b", "group_id": "g"}],
                         "test": [{"code1": "c", "code2": "d", "group_id": "g"}]})


@pytest.mark.parametrize("value", [None, "", "  ", 7, ["tokens"]])
def test_optional_semantic_text_must_be_nonempty_text(tmp_path, value):
    path = tmp_path / "pairs.jsonl"
    path.write_text(json.dumps({"code1": "a", "code2": "b", "semantic_code1": value}), encoding="utf-8")
    with pytest.raises(ValueError, match="semantic_code1 must be nonempty text"):
        read_jsonl(path, require_labels=False)


def test_invalid_json_reports_file_and_line(tmp_path):
    path = tmp_path / "pairs.jsonl"
    path.write_text(chr(10) + "{", encoding="utf-8")
    with pytest.raises(ValueError, match=r"pairs.jsonl:2: invalid JSON"):
        read_jsonl(path)


def test_prediction_inputs_ignore_optional_label_values():
    def tokenizer(*args, **kwargs):
        return {"input_ids": torch.tensor([[0, 2]]), "attention_mask": torch.tensor([[1, 1]])}

    base = {"code1": "a", "code2": "b", "class_support": [0.4, 0.6],
            "ast_relations": [[0, 1]] * 18}
    records = [dict(base), dict(base, label=None), dict(base, label="clone")]
    data = PairDataset(records, tokenizer, 8, "java", 2, include_labels=False)
    batch = next(iter(DataLoader(data, batch_size=3)))
    assert batch["label"].tolist() == [-1, -1, -1]
    torch.testing.assert_close(batch["class_support"], torch.tensor([[0.4, 0.6]] * 3))

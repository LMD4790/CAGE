import json

import pytest
import torch
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

from cage import CAGE
from cage.runtime import load_model, save_model


@pytest.mark.parametrize("ast_features,heads,legacy", [(18, 4, False), (6, 4, False), (18, 8, True)])
def test_checkpoint_restores_architecture_and_predictions(tmp_path, ast_features, heads, legacy):
    vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "<mask>": 4, "a": 5}
    (tmp_path / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    (tmp_path / "merges.txt").write_text("#version: 0.2" + chr(10), encoding="utf-8")
    tokenizer = RobertaTokenizer(
        vocab_file=str(tmp_path / "vocab.json"), merges_file=str(tmp_path / "merges.txt")
    )
    config = RobertaConfig(
        vocab_size=len(tokenizer), hidden_size=32, num_hidden_layers=1,
        num_attention_heads=4, intermediate_size=64, max_position_embeddings=18,
        pad_token_id=1,
    )
    torch.manual_seed(7)
    model = CAGE(RobertaModel(config), 2, ast_features, heads).eval()
    differences = torch.rand(2, ast_features)
    inputs = {
        "input_ids": torch.tensor([[0, 5, 2], [0, 3, 2]]),
        "attention_mask": torch.ones(2, 3, dtype=torch.long),
        "class_support": torch.tensor([[0.4, 0.6], [0.7, 0.3]]),
        "ast_relations": torch.stack((differences, 1 - differences), dim=-1),
    }
    with torch.inference_mode():
        expected = model(**inputs)
    output = tmp_path / "model"
    save_model(model, tokenizer, output, {"language": "java", "max_length": 16})
    if legacy:
        checkpoint = torch.load(output / "best_model.pt", weights_only=True)
        checkpoint.pop("model_config")
        torch.save(checkpoint, output / "best_model.pt")
    restored, _, metadata = load_model(output)
    assert restored.num_ast_features == ast_features
    assert restored.cross_attention.num_heads == heads
    assert metadata == {"language": "java", "max_length": 16}
    with torch.inference_mode():
        torch.testing.assert_close(restored(**inputs), expected, atol=1e-7, rtol=0)

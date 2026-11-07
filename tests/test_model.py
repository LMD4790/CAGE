import torch
import pytest
from transformers import RobertaConfig, RobertaModel

from cage import CAGE


def test_cage_forward_and_attention_shapes():
    config = RobertaConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=8,
        intermediate_size=128,
        max_position_embeddings=66,
        pad_token_id=1,
    )
    model = CAGE(RobertaModel(config), num_classes=3)
    input_ids = torch.randint(2, 127, (2, 24))
    mask = torch.ones_like(input_ids)
    support = torch.softmax(torch.randn(2, 3), dim=-1)
    ast = torch.rand(2, 18, 2)
    logits, weights = model(input_ids, mask, support, ast, return_attention=True)
    assert logits.shape == (2, 3)
    assert weights.shape == (2, 8, 1, 18)
    model.eval()
    with torch.no_grad():
        three_dim = model(input_ids, mask, support, ast)
        flat = torch.cat((ast[:, :, 0], ast[:, :, 1]), dim=1)
        two_dim = model(input_ids, mask, support, flat)
    assert torch.allclose(three_dim, two_dim, atol=1e-6)
    with pytest.raises(ValueError):
        model(input_ids, mask, support[:, :2], ast)

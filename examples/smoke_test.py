#!/usr/bin/env python3
"""Run a data-free forward, backward, and checkpoint smoke test."""

import tempfile
import argparse
from pathlib import Path

import torch
from transformers import RobertaConfig, RobertaModel

from cage import CAGE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.manual_seed(7)
    config = RobertaConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=8,
        intermediate_size=128,
        max_position_embeddings=66,
        pad_token_id=1,
    )
    model = CAGE(RobertaModel(config), num_classes=5).to(args.device)
    model.train()
    input_ids = torch.randint(2, 127, (3, 32))
    input_ids[:, 0] = config.bos_token_id
    attention_mask = torch.ones_like(input_ids)
    class_support = torch.softmax(torch.randn(3, 5), dim=-1)
    ast_relations = torch.rand(3, 18, 2)
    labels = torch.tensor([0, 2, 4])
    ast_relations[:, :, 1] = 1 - ast_relations[:, :, 0]
    input_ids, attention_mask, class_support, ast_relations, labels = [
        value.to(args.device)
        for value in (input_ids, attention_mask, class_support, ast_relations, labels)
    ]

    logits, attention = model(
        input_ids,
        attention_mask,
        class_support,
        ast_relations,
        return_attention=True,
    )
    assert logits.shape == (3, 5)
    assert attention.shape == (3, 8, 1, 18)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.ast_feature_embedding.grad is not None

    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint = Path(temp_dir) / "smoke.pt"
        torch.save(model.state_dict(), checkpoint)
        reloaded = CAGE(RobertaModel(config), num_classes=5).to(args.device)
        reloaded.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True)
        )
        reloaded.eval()
        with torch.inference_mode():
            output = reloaded(input_ids, attention_mask, class_support, ast_relations)
        assert output.shape == (3, 5)

    print(f"CAGE smoke test passed on {args.device}")


if __name__ == "__main__":
    main()

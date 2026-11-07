#!/usr/bin/env python3
"""Check CAGE with a pretrained encoder using a 512-token forward and backward pass."""

import argparse
import json

import torch
from transformers import RobertaModel, RobertaTokenizer

from cage import CAGE
from cage.features import pairwise_ast_relations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="microsoft/codebert-base")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    a = "class A { int inc(int x) { return x + 1; } }"
    b = "class B { int next(int y) { return y + 1; } }"
    torch.manual_seed(7)
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    model = CAGE(RobertaModel.from_pretrained(args.model_name), num_classes=5).to(args.device)
    encoded = tokenizer(
        a, b, padding="max_length", max_length=512, truncation=True, return_tensors="pt"
    )
    encoded = {key: value.to(args.device) for key, value in encoded.items() if key in {"input_ids", "attention_mask"}}
    support = torch.full((1, 5), 0.2, device=args.device)
    ast = torch.tensor(pairwise_ast_relations(a, b), device=args.device).unsqueeze(0)
    model.train()
    logits = model(**encoded, class_support=support, ast_relations=ast)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([1], device=args.device))
    loss.backward()
    for module in (model.encoder, model.ast_value_encoder, model.class_support_fusion, model.cross_attention, model.classifier):
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert logits.shape == (1, 5) and torch.isfinite(loss)
    print(json.dumps({
        "status": "passed", "device": args.device, "input_shape": list(encoded["input_ids"].shape),
        "logit_shape": list(logits.shape), "hidden_size": model.hidden_size,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "finite_gradients": True,
    }))


if __name__ == "__main__":
    main()

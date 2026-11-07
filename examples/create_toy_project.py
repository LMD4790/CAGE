#!/usr/bin/env python3
"""Generate synthetic Java pairs and a small random RoBERTa checkpoint."""

import argparse
import json
from pathlib import Path

import torch
from tokenizers import ByteLevelBPETokenizer
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer


def create_project(output_dir):
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output-dir must be empty")
    raw = output_dir / "raw"
    encoder = output_dir / "encoder"
    raw.mkdir(parents=True, exist_ok=True)
    encoder.mkdir()
    training_code = []
    for split, size in (("train", 30), ("val", 6), ("test", 6)):
        rows = []
        for index in range(size):
            name = f"{split}_{index}"
            code1 = f"int add_{name}(int x) {{ return x + {index + 1}; }}"
            if index % 2:
                code2 = f"int sum_{name}(int y) {{ return y + {index + 1}; }}"
            else:
                code2 = f"int count_{name}(int x) {{ int r=0; while(x>0) {{ r++; x--; }} return r; }}"
            rows.append({"id": name, "code1": code1, "code2": code2, "label": index % 2})
            if split == "train":
                training_code.extend([code1, code2])
        with (raw / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(
        training_code, vocab_size=300, min_frequency=1,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
    )
    bpe.save_model(str(encoder))
    tokenizer = RobertaTokenizer(
        vocab_file=str(encoder / "vocab.json"), merges_file=str(encoder / "merges.txt"),
        model_max_length=512,
    )
    tokenizer.save_pretrained(encoder)
    torch.manual_seed(7)
    config = RobertaConfig(
        vocab_size=len(tokenizer), hidden_size=64, num_hidden_layers=2,
        num_attention_heads=8, intermediate_size=128, max_position_embeddings=514,
        pad_token_id=tokenizer.pad_token_id, bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    RobertaModel(config).save_pretrained(encoder)
    print(f"Synthetic examples and random encoder written to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    create_project(parser.parse_args().output_dir)

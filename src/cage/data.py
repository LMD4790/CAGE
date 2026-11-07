"""JSONL validation and CodeBERT paired-code inputs."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import pairwise_ast_relations


def read_jsonl(path, require_labels=True):
    records = []
    with Path(path).open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number}: invalid JSON: {error.msg}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected a JSON object")
            if isinstance(row.get("original_data"), dict):
                row = {**row["original_data"], **row}
            for key in ("code1", "code2"):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f"{path}:{number}: {key} must be nonempty text")
            for key in ("semantic_code1", "semantic_code2"):
                if key in row and (not isinstance(row[key], str) or not row[key].strip()):
                    raise ValueError(f"{path}:{number}: {key} must be nonempty text when provided")
            if require_labels and (
                type(row.get("label")) is not int or row["label"] < 0
            ):
                raise ValueError(f"{path}:{number}: label must be a nonnegative integer")
            records.append(row)
    if not records:
        raise ValueError(f"{path}: no examples")
    return records


def support_vector(row, num_classes):
    values = np.asarray(
        row.get("class_support", row.get("ml_fusion_score", [])), dtype=np.float32
    )
    if values.shape != (num_classes,):
        raise ValueError(f"Each class_support vector needs {num_classes} entries")
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("class_support must contain finite probabilities in [0, 1]")
    if not np.isclose(values.sum(), 1, atol=1e-5):
        raise ValueError("class_support probabilities must sum to 1")
    return values


IDENTITY_POLICY = "source_tokens_preserve_literals_comments_v1"
OOF_POLICY = "connected_source_and_declared_groups_v1"
_IDENTITY_TOKENS = re.compile(
    r'"""[\s\S]*?"""|"(?:\\[\s\S]|[^"\\])*"|'
    r"'(?:\\[\s\S]|[^'\\])*'|//[^\r\n]*|/\*[\s\S]*?\*/|"
    r"[A-Za-z_$][\w$]*|[0-9][A-Za-z0-9_.]*|"
    r">>>=|>>>=?|<<=|>>=|->\*|\.\.\.|::|->|\.\*|"
    r"\+\+|--|&&|\|\||==|!=|<=|>=|<<|>>|[+*/%&|^!-]=|\S"
)


def code_identity(code):
    """Ignore inter-token spacing, preserving literals, comments and operators."""
    if not isinstance(code, str) or not code.strip():
        raise ValueError("Code identity requires nonempty source text")
    tokens = _IDENTITY_TOKENS.findall(code)
    payload = json.dumps(tokens, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _declared_group(row):
    if "group_id" not in row:
        return None
    group = row["group_id"]
    if type(group) not in (str, int) or not str(group).strip():
        raise ValueError("group_id must be a nonempty string or integer")
    return str(group)


def validate_splits(splits):
    """Reject duplicate pairs, shared source across splits and crossing groups."""
    seen_pairs, seen_groups, seen_codes = {}, {}, {}
    group_presence = ["group_id" in row for rows in splits.values() for row in rows]
    if any(group_presence) and not all(group_presence):
        raise ValueError("Provide group_id for every row in every split, or omit it entirely")
    for split, rows in splits.items():
        for index, row in enumerate(rows):
            identities = [code_identity(row[key]) for key in ("code1", "code2")]
            key = tuple(sorted(identities))
            if key in seen_pairs:
                previous_split, previous_index, previous_label = seen_pairs[key]
                conflict = " with conflicting labels" if previous_label != row.get("label") else ""
                raise ValueError(
                    f"Duplicate code pair{conflict}: {previous_split}[{previous_index}] and {split}[{index}]"
                )
            seen_pairs[key] = (split, index, row.get("label"))
            for identity in identities:
                if identity in seen_codes and seen_codes[identity] != split:
                    raise ValueError(f"Shared source code occurs in {seen_codes[identity]} and {split}")
                seen_codes[identity] = split
            group = _declared_group(row)
            if group is not None:
                if group in seen_groups and seen_groups[group] != split:
                    raise ValueError(f"group_id occurs in {seen_groups[group]} and {split}")
                seen_groups[group] = split


def training_groups(records):
    """Connected components prevent shared source or declared groups crossing OOF folds."""
    validate_splits({"train": records})
    parents = list(range(len(records)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    seen = {}
    for index, row in enumerate(records):
        keys = [("source", code_identity(row[key])) for key in ("code1", "code2")]
        group = _declared_group(row)
        if group is not None:
            keys.append(("group", group))
        for key in keys:
            if key in seen:
                parents[root(index)] = root(seen[key])
            else:
                seen[key] = index
    mapping = {}
    return np.asarray([mapping.setdefault(root(i), len(mapping)) for i in range(len(records))])


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PairDataset(Dataset):
    def __init__(self, records, tokenizer, max_length, language, num_classes, include_labels=True):
        self.rows = []
        self.records = records
        for row in records:
            # Optional semantic text does not replace source used by AST extraction.
            encoded = tokenizer(
                row.get("semantic_code1", row["code1"]),
                row.get("semantic_code2", row["code2"]),
                max_length=max_length,
                truncation="longest_first",
                padding="max_length",
                return_tensors="pt",
            )
            if "ast_relations" in row:
                ast = np.asarray(row["ast_relations"], dtype=np.float32)
            else:
                ast = pairwise_ast_relations(row["code1"], row["code2"], language)
            if ast.shape != (18, 2) or not np.isfinite(ast).all():
                raise ValueError("ast_relations must be a finite [18, 2] array")
            if np.any(ast < 0) or np.any(ast > 1) or not np.allclose(ast.sum(1), 1, atol=1e-5):
                raise ValueError("AST relations must be [difference, 1-difference] in [0, 1]")
            self.rows.append({
                "input_ids": encoded["input_ids"][0],
                "attention_mask": encoded["attention_mask"][0],
                "class_support": torch.from_numpy(support_vector(row, num_classes)),
                "ast_relations": torch.from_numpy(ast),
                "label": row.get("label", -1) if include_labels else -1,
            })

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

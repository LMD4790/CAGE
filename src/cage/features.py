"""Pairwise lexical and Tree-sitter structural feature extraction."""

from __future__ import annotations

import math
import re
import warnings
from collections import Counter
from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np
from rapidfuzz.distance import Levenshtein


AST_FEATURE_NAMES = (
    "max_depth", "log_node_count", "named_node_ratio", "leaf_ratio",
    "mean_children", "log_max_children", "node_type_entropy",
    "node_type_diversity", "branch_density", "loop_density",
    "switch_density", "exception_density", "call_density",
    "assignment_density", "declaration_density", "jump_density",
    "identifier_density", "literal_density",
)
COMPARISON_FEATURE_NAMES = (
    "jaccard", "dice", "overlap", "cosine", "levenshtein", "tfidf_cosine",
    "unique_tokens_code1", "unique_tokens_code2", "common_tokens",
    "total_tokens_code1", "total_tokens_code2", "tokens_c1", "tokens_c2",
    "sim_mean", "sim_std", "sim_max", "sim_min", "token_ratio", "token_diff",
    "struct_inter",
)
_TOKEN_RE = re.compile(
    r"[a-zA-Z_][a-zA-Z0-9_]*|\d+|[+\-*/%&|^~!<>=]=?|"
    r"[(){}\[\];,:.]|&&|\|\||\S+"
)
_LINE_COMMENT_RE = re.compile(r"//.*")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_LITERAL_RE = re.compile(r""""[^"]*"|'[^']*'""")
_NUMBER_RE = re.compile(r"\b\d+\.?\d*\b")


def tokenize_code(code: str) -> list[str]:
    text = _LINE_COMMENT_RE.sub("", code or "")
    text = _BLOCK_COMMENT_RE.sub("", text)
    text = re.sub(r"\s+", " ", text)
    text = _LITERAL_RE.sub(" STRING_LITERAL ", text)
    text = _NUMBER_RE.sub(" NUMBER_LITERAL ", text)
    return [token for token in _TOKEN_RE.findall(text) if token.strip()]


def _cosine(tokens_a: Sequence[str], tokens_b: Sequence[str]) -> float:
    count_a, count_b = Counter(tokens_a), Counter(tokens_b)
    norm_a = math.sqrt(sum(value * value for value in count_a.values()))
    norm_b = math.sqrt(sum(value * value for value in count_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    dot = sum(value * count_b.get(token, 0) for token, value in count_a.items())
    return dot / (norm_a * norm_b)


def pairwise_comparison_features(
    code_a: str, code_b: str, tfidf_cosine: float
) -> np.ndarray:
    """Return the documented 20 LightGBM inputs in a fixed column order."""
    tokens_a, tokens_b = tokenize_code(code_a), tokenize_code(code_b)
    set_a, set_b = set(tokens_a), set(tokens_b)
    overlap = len(set_a & set_b)
    union = len(set_a | set_b)
    smaller = min(len(set_a), len(set_b))
    jaccard = overlap / union if union else 1.0
    dice_denom = len(set_a) + len(set_b)
    dice = 2.0 * overlap / dice_denom if dice_denom else 1.0
    overlap_score = overlap / smaller if smaller else 0.0
    cosine = _cosine(tokens_a, tokens_b)
    normalized_a = " ".join((code_a or "").split())
    normalized_b = " ".join((code_b or "").split())
    longest = max(len(normalized_a), len(normalized_b))
    levenshtein = (
        Levenshtein.normalized_similarity(normalized_a, normalized_b)
        if longest else 1.0
    )
    similarities = np.asarray(
        [jaccard, dice, overlap_score, cosine, levenshtein, tfidf_cosine],
        dtype=np.float64,
    )
    whitespace_a = len((code_a or "").split())
    whitespace_b = len((code_b or "").split())
    values = [
        *similarities.tolist(),
        float(len(set_a)), float(len(set_b)), float(overlap),
        float(len(tokens_a)), float(len(tokens_b)),
        float(whitespace_a), float(whitespace_b),
        float(similarities.mean()), float(similarities.std(ddof=1)),
        float(similarities.max()), float(similarities.min()),
        float(np.clip(whitespace_a / (whitespace_b + 1e-6), 0.0, 10.0)),
        float(abs(whitespace_a - whitespace_b)),
        float(cosine * (1.0 - levenshtein)),
    ]
    if len(values) != 20:
        raise RuntimeError(f"Expected 20 comparison features, got {len(values)}")
    return np.asarray(values, dtype=np.float64)


@lru_cache(maxsize=4)
def _parser(language: str):
    from tree_sitter_languages import get_parser

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"Language\(path, name\) is deprecated")
        return get_parser(language)


def extract_ast_features(code: str, language: str = "java") -> np.ndarray:
    """Extract the 18 global AST statistics for Java or C source code."""
    if language not in {"java", "c"}:
        raise ValueError("The bundled AST statistic mapping supports 'java' and 'c'")
    tree = _parser(language).parse((code or "").encode("utf-8"))
    if tree is None:
        raise ValueError(f"Tree-sitter failed to parse {language} source")
    root = tree.root_node
    node_types: list[str] = []
    child_counts: list[int] = []
    named_count = leaf_count = 0
    max_depth = 1
    stack = [(root, 1)]
    while stack:
        node, depth = stack.pop()
        node_types.append(node.type)
        child_counts.append(len(node.children))
        named_count += int(node.is_named)
        leaf_count += int(len(node.children) == 0)
        max_depth = max(max_depth, depth)
        stack.extend((child, depth + 1) for child in node.children)

    total_nodes = max(len(node_types), 1)
    counts = Counter(node_types)
    probabilities = np.asarray(list(counts.values()), dtype=np.float64) / total_nodes
    entropy = float(-(probabilities * np.log(probabilities + 1e-12)).sum())
    if len(probabilities) > 1:
        entropy /= math.log(len(probabilities))
    else:
        entropy = 0.0

    if language.lower() in {"c", "c11"}:
        branch = {"if_statement", "switch_statement", "conditional_expression"}
        loop = {"for_statement", "while_statement", "do_statement"}
        switch = {"switch_statement", "case_statement"}
        exception: set[str] = set()
        call = {"call_expression"}
        assignment = {"assignment_expression", "update_expression"}
        declaration = {
            "function_definition", "declaration", "init_declarator",
            "parameter_declaration", "type_definition", "struct_specifier",
            "enum_specifier",
        }
        jump = {"return_statement", "break_statement", "continue_statement", "goto_statement"}
        literal = {"number_literal", "char_literal", "string_literal"}
    else:
        branch = {
            "if_statement", "switch_statement", "switch_expression", "conditional_expression"
        }
        loop = {"for_statement", "enhanced_for_statement", "while_statement", "do_statement"}
        switch = {
            "switch_statement", "switch_expression", "switch_block_statement_group",
            "switch_rule", "switch_label",
        }
        exception = {"try_statement", "catch_clause", "finally_clause", "throw_statement"}
        call = {
            "method_invocation", "object_creation_expression", "explicit_constructor_invocation"
        }
        assignment = {"assignment_expression", "update_expression"}
        declaration = {
            "method_declaration", "constructor_declaration", "class_declaration",
            "interface_declaration", "enum_declaration", "field_declaration",
            "local_variable_declaration", "variable_declarator",
        }
        jump = {
            "return_statement", "break_statement", "continue_statement", "yield_statement",
            "throw_statement",
        }
        literal = {
            "decimal_integer_literal", "hex_integer_literal", "binary_integer_literal",
            "octal_integer_literal", "decimal_floating_point_literal",
            "hex_floating_point_literal", "string_literal", "character_literal",
            "true", "false", "null_literal",
        }

    def density(types: Iterable[str]) -> float:
        return sum(counts[node_type] for node_type in types) / total_nodes

    features = np.asarray(
        [
            float(max_depth), float(np.log1p(total_nodes)),
            named_count / total_nodes, leaf_count / total_nodes,
            float(np.mean(child_counts)) if child_counts else 0.0,
            float(np.log1p(max(child_counts))) if child_counts else 0.0,
            entropy, len(counts) / total_nodes, density(branch), density(loop),
            density(switch), density(exception), density(call), density(assignment),
            density(declaration), density(jump), counts["identifier"] / total_nodes,
            density(literal),
        ],
        dtype=np.float64,
    )
    if features.shape != (18,):
        raise RuntimeError("AST feature order or dimension changed")
    return features


def pairwise_ast_relations(
    code_a: str, code_b: str, language: str = "java"
) -> np.ndarray:
    """Return [18, 2] normalized-difference/complement relations."""
    values_a = extract_ast_features(code_a, language)
    values_b = extract_ast_features(code_b, language)
    differences = np.minimum(
        1.0,
        np.abs(values_a - values_b)
        / (np.maximum(np.abs(values_a), np.abs(values_b)) + 1e-6),
    )
    return np.stack((differences, 1.0 - differences), axis=-1).astype(np.float32)

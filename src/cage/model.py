"""CAGE: CodeBERT, class-support fusion, and AST relation attention."""

from __future__ import annotations

import torch
from torch import nn


class CAGE(nn.Module):
    """Fuse a pretrained code encoder with class support and AST relations.

    Args:
        encoder: A Hugging Face encoder such as ``RobertaModel`` loaded from
            ``microsoft/codebert-base``.
        num_classes: Number of target labels. The class-support vector must
            have this width.
        num_ast_features: Number of named AST statistics (18 for the supplied
            Java/C feature extractors).
        num_attention_heads: Number of cross-attention heads.

    Inputs use the RoBERTa pair-tokenization format produced by the matching
    tokenizer. ``ast_relations`` may have shape ``[batch, features, 2]`` with
    ``[difference, similarity]`` in the last dimension, or ``[batch,
    2 * features]`` with all differences followed by all similarities.
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        num_ast_features: int = 18,
        num_attention_heads: int = 8,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if num_ast_features < 1:
            raise ValueError("num_ast_features must be positive")

        hidden_size = int(encoder.config.hidden_size)
        if num_attention_heads < 1 or hidden_size % num_attention_heads:
            raise ValueError(
                "encoder hidden size must be divisible by num_attention_heads"
            )

        self.encoder = encoder
        self.num_classes = num_classes
        self.num_ast_features = num_ast_features
        self.hidden_size = hidden_size

        self.class_support_fusion = nn.Sequential(
            nn.Linear(hidden_size + num_classes, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.ast_value_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.GELU(),
            nn.Linear(64, hidden_size),
        )
        self.ast_feature_embedding = nn.Parameter(
            torch.empty(num_ast_features, hidden_size)
        )
        nn.init.normal_(self.ast_feature_embedding, mean=0.0, std=0.02)
        self.ast_token_norm = nn.LayerNorm(hidden_size)
        self.query_norm = nn.LayerNorm(hidden_size)
        self.cross_attention = nn.MultiheadAttention(
            hidden_size,
            num_heads=num_attention_heads,
            batch_first=True,
            dropout=0.1,
        )
        self.attention_residual_norm = nn.LayerNorm(hidden_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.output_norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def _reshape_ast_relations(self, ast_relations: torch.Tensor) -> torch.Tensor:
        if ast_relations.ndim == 2:
            expected = 2 * self.num_ast_features
            if ast_relations.shape[1] != expected:
                raise ValueError(
                    f"Expected {expected} flattened AST values, got "
                    f"{ast_relations.shape[1]}"
                )
            differences = ast_relations[:, : self.num_ast_features]
            similarities = ast_relations[:, self.num_ast_features :]
            ast_relations = torch.stack((differences, similarities), dim=-1)
        elif ast_relations.ndim != 3 or ast_relations.shape[1:] != (
            self.num_ast_features,
            2,
        ):
            raise ValueError(
                "ast_relations must have shape [batch, features, 2] or "
                "[batch, 2 * features]"
            )
        return ast_relations

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        class_support: torch.Tensor,
        ast_relations: torch.Tensor,
        return_attention: bool = False,
    ):
        if class_support.ndim != 2 or class_support.shape[1] != self.num_classes:
            raise ValueError(
                f"class_support must have shape [batch, {self.num_classes}]"
            )
        ast_relations = self._reshape_ast_relations(ast_relations)
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must have matching [batch, length] shapes")
        if ast_relations.shape[0] != input_ids.shape[0] or class_support.shape[0] != input_ids.shape[0]:
            raise ValueError("input, support, and AST batch sizes do not match")

        encoded = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        semantic = encoded.last_hidden_state[:, 0, :]
        semantic_support = self.class_support_fusion(
            torch.cat((semantic, class_support), dim=-1)
        )
        ast_tokens = self.ast_value_encoder(ast_relations)
        ast_tokens = self.ast_token_norm(
            ast_tokens + self.ast_feature_embedding.unsqueeze(0)
        )
        query = self.query_norm(semantic_support).unsqueeze(1)
        attended, weights = self.cross_attention(
            query,
            ast_tokens,
            ast_tokens,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        hidden = self.attention_residual_norm(
            semantic_support + attended.squeeze(1)
        )
        fused = self.output_norm(hidden + self.feed_forward(hidden))
        logits = self.classifier(fused)
        if return_attention:
            return logits, weights
        return logits

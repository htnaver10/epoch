"""A wide residual MLP for binary fire-presence prediction."""

from __future__ import annotations

import math

import torch
from torch import nn


class _WideResidualBlock(nn.Module):
    """Pre-normalized feed-forward residual block."""

    def __init__(
        self,
        width: int,
        expansion: int,
        dropout: float,
        residual_scale: float,
    ) -> None:
        super().__init__()
        hidden_width = width * expansion
        self.norm = nn.LayerNorm(width)
        self.expand = nn.Linear(width, hidden_width)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.contract = nn.Linear(hidden_width, width)
        self.dropout2 = nn.Dropout(dropout)
        self.residual_scale = residual_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.dropout2(
            self.contract(self.dropout1(self.activation(self.expand(self.norm(x)))))
        )
        return x + residual * self.residual_scale


class WideFireMLP(nn.Module):
    """Wide residual MLP returning one binary-classification logit per row.

    The constructor arguments are also the complete architecture description, so a
    saved ``architecture`` dictionary can be used as ``WideFireMLP(**architecture)``
    when restoring a checkpoint.
    """

    def __init__(
        self,
        feature_count: int = 96,
        width: int = 4096,
        expansion: int = 2,
        blocks: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if feature_count <= 0:
            raise ValueError("feature_count must be positive")
        if width <= 0:
            raise ValueError("width must be positive")
        if expansion <= 0:
            raise ValueError("expansion must be positive")
        if blocks <= 0:
            raise ValueError("blocks must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.feature_count = int(feature_count)
        self.width = int(width)
        self.expansion = int(expansion)
        self.blocks_count = int(blocks)
        self.dropout_rate = float(dropout)

        self.input_projection = nn.Linear(self.feature_count, self.width)
        residual_scale = 1.0 / math.sqrt(self.blocks_count)
        self.blocks = nn.ModuleList(
            _WideResidualBlock(
                width=self.width,
                expansion=self.expansion,
                dropout=self.dropout_rate,
                residual_scale=residual_scale,
            )
            for _ in range(self.blocks_count)
        )
        self.output_norm = nn.LayerNorm(self.width)
        self.head = nn.Linear(self.width, 1)

    @property
    def architecture(self) -> dict[str, int | float]:
        """Return constructor-compatible architecture metadata."""
        return {
            "feature_count": self.feature_count,
            "width": self.width,
            "expansion": self.expansion,
            "blocks": self.blocks_count,
            "dropout": self.dropout_rate,
        }

    def initialize_prior(self, prevalence: float) -> None:
        """Set the output bias to the log-odds implied by positive prevalence.

        The rest of the model remains randomly initialized, allowing learning to
        begin immediately. The intercept reflects the class balance; random
        input-dependent activations can shift individual initial predictions.
        """
        prevalence = float(prevalence)
        if not 0.0 < prevalence < 1.0:
            raise ValueError("prevalence must be strictly between 0 and 1")
        with torch.no_grad():
            self.head.bias.fill_(math.log(prevalence / (1.0 - prevalence)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits with the final singleton dimension removed."""
        if x.shape[-1] != self.feature_count:
            raise ValueError(
                f"expected {self.feature_count} input features, got {x.shape[-1]}"
            )
        x = self.input_projection(x)
        for block in self.blocks:
            x = block(x)
        return self.head(self.output_norm(x)).squeeze(-1)

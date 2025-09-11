"""
Transformer Model for RL Agent Policy (v3.0 - Production Stable)

v3.0: Upgraded to a robust, professional-grade architecture with Layer
      Normalization and Dropout to prevent numerical instability (NaN errors).
"""
import torch
import torch.nn as nn

class TransformerSelector(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super(TransformerSelector, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, 
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        embedded_src = self.embedding(src)
        seq_input = embedded_src.unsqueeze(1)
        encoded_output = self.transformer_encoder(seq_input)
        squeezed_output = encoded_output.squeeze(1)
        features = self.layer_norm(squeezed_output)
        return features
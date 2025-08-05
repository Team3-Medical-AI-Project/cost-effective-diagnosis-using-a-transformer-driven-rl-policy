"""
Transformer Model for RL Agent Policy

Description:
This file defines the architecture for the Transformer-based selector, which will
serve as the policy network for the reinforcement learning agent. It uses an
encoder-only architecture, as proposed, to process the combined patient state 
and decide on the next action (either request a test panel or make a diagnosis).
"""

import torch
import torch.nn as nn
import math

class MultiHeadSelfAttention(nn.Module):
    """
    Implementation of the Multi-Head Self-Attention mechanism.
    This module allows the model to jointly attend to information from different 
    representation subspaces at different positions.
    """
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Linear layers for transforming inputs into query, key, and value vectors
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        
        # Final linear layer to project the concatenated heads back to embed_dim
        self.W_o = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        batch_size, seq_length, _ = x.shape
        
        # 1. Linearly project the input to get queries, keys, and values.
        queries = self.W_q(x)
        keys = self.W_k(x)
        values = self.W_v(x)
        
        # 2. Reshape and transpose for multi-head attention.
        # Shape becomes: (batch_size, num_heads, seq_length, head_dim)
        queries = queries.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 3. Compute scaled dot-product attention.
        # Attention(Q, K, V) = softmax( (Q * K^T) / sqrt(d_k) ) * V
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float("-1e20"))
            
        attention_probs = torch.softmax(attention_scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attention_probs, values)
        
        # 4. Concatenate heads and pass through the final linear layer.
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_dim)
        output = self.W_o(context)
        
        return output

class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the Transformer encoder, consisting of a multi-head
    self-attention mechanism followed by a position-wise feed-forward network.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 1. Self-attention sub-layer
        attention_output = self.attention(x, mask)
        # 2. Add & Norm (residual connection)
        x = self.norm1(x + self.dropout(attention_output))
        
        # 3. Feed-forward sub-layer
        ff_output = self.feed_forward(x)
        # 4. Add & Norm (residual connection)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class TransformerSelector(nn.Module):
    """
    The full Transformer model that acts as the RL policy network.
    It takes the combined patient state and outputs action probabilities.
    """
    def __init__(self, input_dim, num_actions, num_layers=2, embed_dim=128, num_heads=4, ff_dim=256, dropout=0.1):
        super(TransformerSelector, self).__init__()
        
        # Layer to project the flat input patient state into the Transformer's embedding dimension
        self.input_embedding = nn.Linear(input_dim, embed_dim)
        
        # A stack of Transformer Encoder Layers
        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )
        
        # The final output layer that maps the Transformer's output to the action space
        # This is the "policy head" of our RL agent.
        self.policy_head = nn.Linear(embed_dim, num_actions)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # The input `x` is a flat vector. The Transformer expects a sequence.
        # We treat the input as a "sequence" of length 1.
        # Shape of x: (batch_size, input_dim)
        
        # 1. Pass the input patient state through the embedding layer.
        # Shape becomes: (batch_size, embed_dim)
        embeddings = self.dropout(self.input_embedding(x))
        
        # 2. Add a "sequence length" dimension of 1.
        # Shape becomes: (batch_size, 1, embed_dim)
        seq_embeddings = embeddings.unsqueeze(1)
        
        # 3. Pass the embeddings through the stack of encoder layers.
        for layer in self.encoder_layers:
            seq_embeddings = layer(seq_embeddings)
            
        # 4. Remove the sequence length dimension to get the final representation.
        # Shape becomes: (batch_size, embed_dim)
        final_representation = seq_embeddings.squeeze(1)
        
        # 5. Use the final representation to produce action probabilities.
        # The output logits are passed to a Softmax in the RL algorithm to get probabilities.
        action_logits = self.policy_head(final_representation)
        
        return action_logits

if __name__ == '__main__':
    # This block is for testing the model architecture to ensure it works as expected.
    print("--- Testing TransformerSelector Architecture ---")
    
    # Define parameters based on our project's data structure
    BATCH_SIZE = 32
    NUM_FEATURES = 35         # From our preprocessed data
    CLASSIFIER_OUTPUT_DIM = 2 # Probabilities for class 0 and 1
    MASK_DIM = 35             # One for each feature
    
    # The total input dimension for the Transformer
    INPUT_DIM = NUM_FEATURES + CLASSIFIER_OUTPUT_DIM + MASK_DIM
    
    # The number of possible actions (e.g., 4 panels + 1 diagnose action)
    NUM_ACTIONS = 5
    
    # Create a dummy input tensor representing a batch of combined patient states
    dummy_patient_state = torch.randn(BATCH_SIZE, INPUT_DIM)
    
    # Initialize the selector
    selector = TransformerSelector(input_dim=INPUT_DIM, num_actions=NUM_ACTIONS)
    
    # Perform a forward pass
    action_logits = selector(dummy_patient_state)
    
    print("Transformer Selector Initialized Successfully!")
    print(f"Number of features: {NUM_FEATURES}")
    print(f"Classifier output dim: {CLASSIFIER_OUTPUT_DIM}")
    print(f"Mask dim: {MASK_DIM}")
    print("-" * 20)
    print(f"Total Input Dimension: {INPUT_DIM}")
    print(f"Input tensor shape: {dummy_patient_state.shape}")
    print(f"Output tensor shape (action logits): {action_logits.shape}")
    print(f"Expected output shape: ({BATCH_SIZE}, {NUM_ACTIONS})")
    
    assert action_logits.shape == (BATCH_SIZE, NUM_ACTIONS)
    print("\nTest PASSED: Output shape is correct.")

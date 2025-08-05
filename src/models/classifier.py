"""
Preliminary Classifier for the RL Agent

Description:
This file defines the architecture for a simple Multi-Layer Perceptron (MLP)
that serves as the preliminary classifier. 

As per the project's flowchart, this model takes the imputed patient state and 
outputs a probability distribution over the possible diagnoses (e.g., survive vs. die).
This output is then concatenated with the patient state and mask to form the 
full input for the Transformer Selector.
"""

import torch
import torch.nn as nn

class PreliminaryClassifier(nn.Module):
    """
    A simple MLP to provide an initial diagnosis probability.
    """
    def __init__(self, input_dim, hidden_dim=128, output_dim=2):
        super(PreliminaryClassifier, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
            # Note: We output raw logits. The Softmax for probabilities
            # will be applied by the loss function during training for
            # better numerical stability.
        )

    def forward(self, x):
        """
        x: The imputed patient state vector.
        Shape: (batch_size, input_dim)
        """
        logits = self.model(x)
        return logits

if __name__ == '__main__':
    # This block serves as a unit test to verify the model's architecture.
    print("--- Testing PreliminaryClassifier Architecture ---")
    
    # Define parameters based on our project's data structure
    BATCH_SIZE = 32
    NUM_FEATURES = 35 # From our preprocessed data
    
    # The number of output classes (e.g., survive vs. die)
    NUM_CLASSES = 2
    
    # Create a dummy input tensor representing a batch of imputed patient states
    dummy_imputed_state = torch.randn(BATCH_SIZE, NUM_FEATURES)
    
    # Initialize the classifier
    classifier = PreliminaryClassifier(input_dim=NUM_FEATURES, output_dim=NUM_CLASSES)
    
    # Perform a forward pass
    output_logits = classifier(dummy_imputed_state)
    
    print("Classifier Initialized Successfully!")
    print(f"Input tensor shape: {dummy_imputed_state.shape}")
    print(f"Output tensor shape (logits): {output_logits.shape}")
    print(f"Expected output shape: ({BATCH_SIZE}, {NUM_CLASSES})")
    
    assert output_logits.shape == (BATCH_SIZE, NUM_CLASSES)
    print("\nTest PASSED: Output shape is correct.")

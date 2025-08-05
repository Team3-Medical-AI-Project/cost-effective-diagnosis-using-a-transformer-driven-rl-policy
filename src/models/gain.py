"""
GAIN: Generative Adversarial Imputation Networks - Model Architecture

Description:
This file defines the architecture for the Generator and Discriminator networks
that form the GAIN model. This is the "blueprint" file that will be imported
by the training scripts.
"""

import torch
import torch.nn as nn

class Generator(nn.Module):
    """
    The Generator network.
    Takes a data vector + mask vector as input and outputs an imputed data vector.
    """
    def __init__(self, input_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim) # No final activation, as data is standardized
        )

    def forward(self, x, m):
        """
        x: data vector with missing values (NaNs replaced by 0)
        m: mask vector (0 for missing, 1 for present)
        """
        input_cat = torch.cat([x, m], dim=1)
        imputed_data = self.model(input_cat)
        return imputed_data

class Discriminator(nn.Module):
    """
    The Discriminator network.
    Takes an imputed data vector + hint vector and outputs a probability mask.
    """
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid() # Sigmoid to output probabilities
        )

    def forward(self, x, h):
        """
        x: the imputed data vector from the Generator
        h: the hint vector
        """
        input_cat = torch.cat([x, h], dim=1)
        probability_mask = self.model(input_cat)
        return probability_mask

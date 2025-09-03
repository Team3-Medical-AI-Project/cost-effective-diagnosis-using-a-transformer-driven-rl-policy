import torch
import torch.nn as nn

class Generator(nn.Module):
    """
    G(x, m) -> imputed data in [0,1] (assumes features scaled to [0,1])
    """
    def __init__(self, input_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, input_dim),
            nn.Sigmoid()  # keep generator output in [0,1]
        )

    def forward(self, x, m):
        return self.net(torch.cat([x, m], dim=1))


class Discriminator(nn.Module):
    """
    D(x_hat, h):
      - default (use_logits=False): returns probabilities in [0,1] (old behavior; BCELoss)
      - use_logits=True: returns logits (new stable path; BCEWithLogitsLoss)
    """
    def __init__(self, input_dim: int, hidden: int = 256, use_logits: bool = False):
        super().__init__()
        self.use_logits = use_logits
        self.backbone = nn.Sequential(
            nn.Linear(input_dim * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, input_dim)  # logits
        )

    def forward(self, x, h):
        logits = self.backbone(torch.cat([x, h], dim=1))
        if self.use_logits:
            return logits
        else:
            return torch.sigmoid(logits)



class DiscriminatorLogits(Discriminator):
    def __init__(self, input_dim: int, hidden: int = 256):
        super().__init__(input_dim=input_dim, hidden=hidden, use_logits=True)

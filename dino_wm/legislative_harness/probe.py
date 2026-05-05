import torch
import torch.nn as nn

class LegislativeProbe(nn.Module):
    def __init__(self, input_dim=255, output_dim=1): # We still need to figure out the input dim
        super().__init__()
        self.W = nn.Linear(input_dim, output_dim)

    def forward(self, z):
        out = self.W(z)
        return out

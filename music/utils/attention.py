import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Learnable weights to decide importance of each time step
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x shape: (batch, time, input_dim)
        # Calculate weights
        weights = self.attention(x) # (batch, time, 1) si hay 12 fragmentos será un vector [num_songs, puntuación_importancia_fragmento, 1]
        weights = F.softmax(weights, dim=1) # Normalize weights to sum to 1 --> [num_songs, probabilidad_importancia_fragmento]
        
        # Weighted sum
        weighted_embedding = torch.sum(x * weights, dim=1) # (batch, input_dim)
        return weighted_embedding
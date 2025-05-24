from torch import nn

class FeedForward(nn.Module):
    def __init__(self, embedding_dim = 128):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.mlp = nn.Sequential([
            nn.Linear(embedding_dim, 3 * embedding_dim), # Hidden layer
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(3 * embedding_dim, embedding_dim),
            nn.Dropout(0.1),
        ])
        self.norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, x):
        return self.norm(x + self.mlp(x)) # post add-norm        
        

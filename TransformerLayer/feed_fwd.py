from torch import Tensor, nn

class FeedForward(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 3 * embedding_dim), # Hidden 
            nn.GELU(), # Like ReLU, but keeps some negatives
            nn.Dropout(0.1), # Prevents overfitting, could adjust p value
            nn.Linear(3 * embedding_dim, embedding_dim), # Output 
            nn.Dropout(0.1),
        )
        self.norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, x: Tensor):
        return self.norm(x + self.mlp(x)) # post add-norm        
        

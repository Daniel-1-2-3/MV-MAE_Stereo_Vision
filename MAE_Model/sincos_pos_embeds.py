import numpy as np

class PosEmbed():
    
    @staticmethod
    def get_2d_sincos_pos_embed(embed_dim, grid_h_size, grid_w_size):
        """
        Args:
            embed_dim (int): Embedding dimension for each patch
            grid_h_size (int), grid_w-size(int): Number of patches vertically and horizontally
        """
        grid_h = np.arange(grid_h_size, dtype=np.float32)
        grid_w = np.arange(grid_w_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)
        grid = np.stack(grid, axis=0)

        grid = grid.reshape([2, 1, grid_h_size, grid_w_size])
        pos_embed = PosEmbed.get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
        return pos_embed # (grid_h_size * grid_w_size, embed_dim)
    
    @staticmethod
    def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
        """
        Args:
            embed_dim (int): Embedding dimension for each patch
            grid (numpy array): Array of shape (2, height, width), where grid[0] 
                                is the x_coordinates of all cells in the grid, while
                                grid[1] is the y_coordinates of all cells in the grid
        """
        assert embed_dim % 2 == 0
            
        embed_h = PosEmbed.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0]) # (M, embed_dim / 2)
        embed_w = PosEmbed.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1]) # (M, embed_dim / 2)
        emb = np.concatenate([embed_h, embed_w], axis=1) # (M, embed_dim)
        return emb
    
    @staticmethod
    def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
        """
        Args:
            embed_dim (int): Embedding dimension for each patch
            pos (numpy array):  A list of positions to be encoded (H, W), could be
                                x or y coordinates
        """
        assert embed_dim % 2  == 0
        # Frequency spectrum
        omega = np.arange(embed_dim // 2, dtype=np.float32)
        omega /= embed_dim / 2.0
        omega = 1.0 / 10000 ** omega 
        
        pos = pos.reshape(-1) # Turn pos is a flat array (M = H * W)
        out = np.einsum("m,d->md", pos, omega)  # (M, embed_dim / 2), outer product

        emb_sin = np.sin(out) # (M, embed_dim / 2)
        emb_cos = np.cos(out) # (M, embed_dim / 2)

        emb = np.concatenate([emb_sin, emb_cos], axis=1) # (M, embed_dim)
        return emb 
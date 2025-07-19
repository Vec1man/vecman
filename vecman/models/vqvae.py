"""VECMAN VQ-VAE model implementation for text embeddings."""

import torch
from torch import nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_codes: int, code_dim: int, beta: float = 0.25):
        """Vector Quantizer module for VQ-VAE.
        
        Args:
            num_codes: Number of codes in the codebook
            code_dim: Dimension of each code vector
            beta: Commitment loss coefficient
        """
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.codebook = nn.Parameter(torch.randn(num_codes, code_dim) * 0.1)
        self.beta = beta
    
    def forward(self, z):
        """Forward pass through vector quantizer.
        
        Args:
            z: Input tensor of shape (batch_size, code_dim)
            
        Returns:
            Tuple of (quantized_z, indices, vq_loss)
        """
        # Ensure input has correct shape
        if z.dim() == 1:
            z = z.unsqueeze(0)
        
        batch_size, dim = z.shape
        assert dim == self.code_dim, f"Input dimension {dim} doesn't match code dimension {self.code_dim}"
        
        # Compute distances to codebook vectors
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z*e
        z_squared = torch.sum(z.pow(2), dim=1, keepdim=True)  # (batch_size, 1)
        codebook_squared = torch.sum(self.codebook.pow(2), dim=1)  # (num_codes,)
        distances = z_squared + codebook_squared.unsqueeze(0) - 2 * torch.matmul(z, self.codebook.t())
        
        # Find closest codebook vectors
        idx = torch.argmin(distances, dim=1)  # (batch_size,)
        z_q = self.codebook[idx]  # (batch_size, code_dim)
        
        # Compute VQ loss
        commitment_loss = F.mse_loss(z.detach(), z_q)
        codebook_loss = F.mse_loss(z, z_q.detach())
        vq_loss = codebook_loss + self.beta * commitment_loss
        
        # Straight-through estimator
        z_q = z + (z_q - z).detach()
        
        return z_q, idx, vq_loss

class Encoder(nn.Module):
    def __init__(self, d_in: int, hidden: int, d_lat: int):
        """Encoder network for VQ-VAE.
        
        Args:
            d_in: Input dimension
            hidden: Hidden layer dimension
            d_lat: Latent dimension
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, d_lat)
        )
    
    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, d_lat: int, hidden: int, d_out: int):
        """Decoder network for VQ-VAE.
        
        Args:
            d_lat: Latent dimension
            hidden: Hidden layer dimension
            d_out: Output dimension
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_lat, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, d_out)
        )
    
    def forward(self, z):
        return self.net(z)

class VQVAE(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 1024, latent_bits: int = 16):
        """VQ-VAE model for text embeddings.
        
        Args:
            input_dim: Input embedding dimension
            hidden: Hidden layer dimension
            latent_bits: Number of bits for latent representation
        """
        super().__init__()
        # Fix: Use a more reasonable latent dimension calculation
        # Instead of latent_bits // 8, use a minimum of 64 dimensions
        # This ensures the latent space is expressive enough
        lat_dim = max(64, input_dim // 4, latent_bits * 4)
        
        self.input_dim = input_dim
        self.latent_bits = latent_bits
        self.lat_dim = lat_dim
        
        # Validate inputs
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if latent_bits <= 0:
            raise ValueError(f"latent_bits must be positive, got {latent_bits}")
        
        self.encoder = Encoder(input_dim, hidden, lat_dim)
        self.vq = VectorQuantizer(2 ** latent_bits, lat_dim)
        self.decoder = Decoder(lat_dim, hidden, input_dim)
    
    def forward(self, x):
        """Forward pass through VQ-VAE.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tuple of (reconstruction, indices, total_loss, reconstruction_loss)
        """
        # Validate input
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        batch_size, dim = x.shape
        if dim != self.input_dim:
            raise ValueError(f"Input dimension {dim} doesn't match expected {self.input_dim}")
        
        # Encode
        z = self.encoder(x)
        
        # Quantize
        z_q, idx, vq_loss = self.vq(z)
        
        # Decode
        recon = self.decoder(z_q)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(recon, x)
        
        # Total loss
        total_loss = recon_loss + vq_loss
        
        return recon, idx, total_loss, recon_loss 
"""VECMAN v3 models: product-quantized VQ-VAE for embedding compression.

The model compresses dense embeddings (e.g. 384-dim sentence-transformer
vectors) into a small number of discrete codes:

    input (D) --encoder--> latent (d) --product quantizer--> M codes
                                                             (one per subspace)

With the defaults (M=8 subquantizers, 256 codes each) a document is stored
as 8 bytes instead of D*4 bytes of float32 (192x smaller for D=384).

Codebooks are trained with exponential-moving-average (EMA) updates and
dead-code reinitialization, the standard remedies for codebook collapse.
"""

from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def codes_dtype(codes_per_subquantizer: int) -> np.dtype:
    """Smallest unsigned integer dtype that can hold every codebook index."""
    if codes_per_subquantizer <= 2 ** 8:
        return np.dtype(np.uint8)
    if codes_per_subquantizer <= 2 ** 16:
        return np.dtype(np.uint16)
    return np.dtype(np.uint32)


class EMAVectorQuantizer(nn.Module):
    """Single-codebook vector quantizer with EMA updates and dead-code reinit.

    Args:
        num_codes: Number of entries in the codebook.
        code_dim: Dimension of each code vector.
        beta: Commitment loss coefficient.
        decay: EMA decay for codebook updates.
        eps: Laplace-smoothing epsilon for cluster sizes.
    """

    def __init__(self, num_codes: int, code_dim: int, beta: float = 0.25,
                 decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        if num_codes <= 0 or code_dim <= 0:
            raise ValueError("num_codes and code_dim must be positive")
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.beta = beta
        self.decay = decay
        self.eps = eps
        embed = torch.randn(num_codes, code_dim) * 0.1
        self.register_buffer("codebook", embed)
        self.register_buffer("ema_cluster_size", torch.zeros(num_codes))
        self.register_buffer("ema_embed_sum", embed.clone())

    @torch.no_grad()
    def nearest(self, z: torch.Tensor) -> torch.Tensor:
        distances = (
            z.pow(2).sum(1, keepdim=True)
            + self.codebook.pow(2).sum(1)
            - 2 * z @ self.codebook.t()
        )
        return distances.argmin(1)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if z.dim() == 1:
            z = z.unsqueeze(0)
        idx = self.nearest(z)
        z_q = self.codebook[idx]

        if self.training:
            self._ema_update(z, idx)
            self._reinit_dead_codes(z)

        commitment_loss = F.mse_loss(z, z_q.detach())
        loss = self.beta * commitment_loss
        # Straight-through estimator: gradients flow to the encoder as if
        # quantization were the identity.
        z_q = z + (z_q - z).detach()
        return z_q, idx, loss

    @torch.no_grad()
    def _ema_update(self, z: torch.Tensor, idx: torch.Tensor) -> None:
        one_hot = F.one_hot(idx, self.num_codes).type(z.dtype)
        self.ema_cluster_size.mul_(self.decay).add_(one_hot.sum(0), alpha=1 - self.decay)
        self.ema_embed_sum.mul_(self.decay).add_(one_hot.t() @ z, alpha=1 - self.decay)
        n = self.ema_cluster_size.sum()
        smoothed = (self.ema_cluster_size + self.eps) / (n + self.num_codes * self.eps) * n
        self.codebook.copy_(self.ema_embed_sum / smoothed.unsqueeze(1))

    @torch.no_grad()
    def _reinit_dead_codes(self, z: torch.Tensor) -> None:
        dead = self.ema_cluster_size < 1e-3
        n_dead = int(dead.sum())
        if n_dead == 0 or z.shape[0] == 0:
            return
        rand = z[torch.randint(0, z.shape[0], (n_dead,), device=z.device)]
        self.codebook[dead] = rand
        self.ema_embed_sum[dead] = rand
        self.ema_cluster_size[dead] = 1.0


class ProductQuantizer(nn.Module):
    """Product quantizer: splits the latent into M subspaces, each with its
    own EMA codebook. A vector is represented by M small integers.
    """

    def __init__(self, code_dim: int, num_subquantizers: int = 8,
                 codes_per_subquantizer: int = 256, beta: float = 0.25,
                 decay: float = 0.99):
        super().__init__()
        if code_dim % num_subquantizers != 0:
            raise ValueError(
                f"code_dim ({code_dim}) must be divisible by "
                f"num_subquantizers ({num_subquantizers})"
            )
        self.code_dim = code_dim
        self.num_subquantizers = num_subquantizers
        self.codes_per_subquantizer = codes_per_subquantizer
        self.sub_dim = code_dim // num_subquantizers
        self.quantizers = nn.ModuleList(
            EMAVectorQuantizer(codes_per_subquantizer, self.sub_dim, beta, decay)
            for _ in range(num_subquantizers)
        )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        chunks = z.chunk(self.num_subquantizers, dim=1)
        z_qs, idxs = [], []
        loss = z.new_zeros(())
        for quantizer, chunk in zip(self.quantizers, chunks):
            z_q, idx, sub_loss = quantizer(chunk)
            z_qs.append(z_q)
            idxs.append(idx)
            loss = loss + sub_loss
        return torch.cat(z_qs, dim=1), torch.stack(idxs, dim=1), loss / self.num_subquantizers

    @torch.no_grad()
    def encode_indices(self, z: torch.Tensor) -> torch.Tensor:
        """Latents (B, code_dim) -> codes (B, M)."""
        chunks = z.chunk(self.num_subquantizers, dim=1)
        return torch.stack(
            [q.nearest(c) for q, c in zip(self.quantizers, chunks)], dim=1
        )

    @torch.no_grad()
    def decode_indices(self, idx: torch.Tensor) -> torch.Tensor:
        """Codes (B, M) -> reconstructed latents (B, code_dim)."""
        if idx.dim() == 1:
            idx = idx.unsqueeze(0)
        parts = [
            q.codebook[idx[:, m].long()]
            for m, q in enumerate(self.quantizers)
        ]
        return torch.cat(parts, dim=1)


def _mlp(d_in: int, hidden: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_in, hidden),
        nn.LayerNorm(hidden),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden, hidden),
        nn.LayerNorm(hidden),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden, d_out),
    )


def default_latent_dim(input_dim: int, num_subquantizers: int) -> int:
    lat = max(64, input_dim // 4)
    # Round up to a multiple of the number of subquantizers.
    remainder = lat % num_subquantizers
    if remainder:
        lat += num_subquantizers - remainder
    return lat


class VQVAE(nn.Module):
    """Product-quantized VQ-VAE for embedding compression.

    Args:
        input_dim: Dimension of the input embeddings.
        hidden: Hidden layer width of the encoder/decoder MLPs.
        latent_dim: Latent dimension (default: max(64, input_dim // 4),
            rounded up to a multiple of num_subquantizers).
        num_subquantizers: Number of PQ subspaces (M). A document is stored
            as M integers.
        codes_per_subquantizer: Codebook size per subspace (K). K <= 256
            keeps storage at one byte per code.
        beta: Commitment loss coefficient.
        sim_weight: Weight of the similarity-preservation loss, which trains
            the latent space so that pairwise cosine similarities match those
            of the inputs. Without it the encoder can satisfy reconstruction
            while collapsing all latents onto one direction, destroying
            retrieval quality.
    """

    def __init__(self, input_dim: int, hidden: int = 1024,
                 latent_dim: Optional[int] = None,
                 num_subquantizers: int = 8,
                 codes_per_subquantizer: int = 256,
                 beta: float = 0.25,
                 sim_weight: float = 1.0):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if num_subquantizers <= 0:
            raise ValueError(f"num_subquantizers must be positive, got {num_subquantizers}")
        if codes_per_subquantizer <= 1:
            raise ValueError(
                f"codes_per_subquantizer must be > 1, got {codes_per_subquantizer}"
            )
        if latent_dim is None:
            latent_dim = default_latent_dim(input_dim, num_subquantizers)
        if latent_dim % num_subquantizers != 0:
            raise ValueError(
                f"latent_dim ({latent_dim}) must be divisible by "
                f"num_subquantizers ({num_subquantizers})"
            )

        self.input_dim = input_dim
        self.hidden = hidden
        self.lat_dim = latent_dim
        self.num_subquantizers = num_subquantizers
        self.codes_per_subquantizer = codes_per_subquantizer
        self.sim_weight = sim_weight

        self.encoder = _mlp(input_dim, hidden, latent_dim)
        # Center/scale each latent dimension before quantization. Without
        # this, a shared offset in the encoder output dominates the
        # nearest-code assignment and collapses most documents onto the
        # same codes.
        self.pre_vq_norm = nn.BatchNorm1d(latent_dim, affine=False)
        self.vq = ProductQuantizer(
            latent_dim, num_subquantizers, codes_per_subquantizer, beta
        )
        self.decoder = _mlp(latent_dim, hidden, input_dim)

    @property
    def codes_dtype(self) -> np.dtype:
        return codes_dtype(self.codes_per_subquantizer)

    def forward(self, x: torch.Tensor):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Input dimension {x.shape[1]} doesn't match expected {self.input_dim}"
            )
        z = self.pre_vq_norm(self.encoder(x))
        z_q, idx, vq_loss = self.vq(z)
        recon = self.decoder(z_q)
        recon_loss = F.mse_loss(recon, x)
        total_loss = recon_loss + vq_loss
        if self.sim_weight > 0 and x.shape[0] > 1:
            total_loss = total_loss + self.sim_weight * self._similarity_loss(x, z)
        return recon, idx, total_loss, recon_loss

    @staticmethod
    def _similarity_loss(x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Match the latent pairwise-cosine structure to the input's.

        Latents are centered on the batch mean first: a shared offset is
        harmless for reconstruction but dominates cosine similarity, so it
        is removed before comparing geometries.
        """
        x_n = F.normalize(x - x.mean(dim=0, keepdim=True), dim=1, eps=1e-8)
        z_n = F.normalize(z - z.mean(dim=0, keepdim=True), dim=1, eps=1e-8)
        return F.mse_loss(z_n @ z_n.t(), x_n @ x_n.t())

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Embeddings (B, input_dim) -> continuous latents (B, lat_dim).

        Uses the pre-VQ norm's running statistics, so call in eval mode.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        was_training = self.training
        self.eval()
        try:
            return self.pre_vq_norm(self.encoder(x))
        finally:
            self.train(was_training)

    @torch.no_grad()
    def compress(self, x: torch.Tensor) -> np.ndarray:
        """Embeddings (B, input_dim) -> codes (B, M) as the smallest safe dtype."""
        z = self.encode(x)
        idx = self.vq.encode_indices(z)
        return idx.cpu().numpy().astype(self.codes_dtype)

    @torch.no_grad()
    def decompress(self, codes: np.ndarray) -> torch.Tensor:
        """Codes (B, M) -> reconstructed latents (B, lat_dim)."""
        idx = torch.from_numpy(np.ascontiguousarray(codes).astype(np.int64))
        return self.vq.decode_indices(idx)

    def config(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden,
            "latent_dim": self.lat_dim,
            "num_subquantizers": self.num_subquantizers,
            "codes_per_subquantizer": self.codes_per_subquantizer,
            "sim_weight": self.sim_weight,
        }

    @classmethod
    def from_config(cls, meta: dict) -> "VQVAE":
        return cls(
            input_dim=meta["input_dim"],
            hidden=meta.get("hidden_dim", 1024),
            latent_dim=meta.get("latent_dim"),
            num_subquantizers=meta.get("num_subquantizers", 8),
            codes_per_subquantizer=meta.get("codes_per_subquantizer", 256),
            sim_weight=meta.get("sim_weight", 1.0),
        )

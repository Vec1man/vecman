"""VECMAN training utilities for the product-quantized VQ-VAE."""

import copy
import json
import os
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from ..models.vqvae import VQVAE


class NPZStreamDataset(IterableDataset):
    """Streams batches from a .npy embedding matrix via memory-mapping, so
    corpora far larger than RAM can be trained on."""

    def __init__(self, np_file: str, batch_size: int, input_dim: int):
        super().__init__()
        self.path = Path(np_file)
        self.bs = batch_size
        self.d = input_dim

        if not self.path.exists():
            raise FileNotFoundError(f"Corpus file not found: {self.path}")
        arr = np.load(self.path, mmap_mode="r")
        if arr.ndim != 2 or arr.shape[1] != input_dim:
            raise ValueError(
                f"Corpus {self.path} has shape {arr.shape}, expected (N, {input_dim})"
            )

    def __iter__(self):
        arr = np.load(self.path, mmap_mode="r")
        idx, n = 0, arr.shape[0]
        while idx < n:
            batch = np.nan_to_num(
                arr[idx:idx + self.bs], nan=0.0, posinf=1.0, neginf=-1.0
            )
            yield torch.from_numpy(np.ascontiguousarray(batch)).float()
            idx += self.bs


def resolve_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA requested but not available, falling back to CPU", stacklevel=2)
        return "cpu"
    return device


def _train_loop(model: VQVAE, loader: DataLoader, epochs: int, device: str,
                learning_rate: float = 1e-3) -> dict:
    """Train the model in place; returns the best (lowest-loss) state dict."""
    device = resolve_device(device)
    model.to(device)
    model.train()

    opt = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.8, patience=2
    )

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())

    for ep in range(1, epochs + 1):
        epoch_loss, num_samples = 0.0, 0
        with tqdm(loader, desc=f"Epoch {ep}/{epochs}") as pbar:
            for batch in pbar:
                batch = batch.to(device)
                _, _, total_loss, recon_loss = model(batch)
                opt.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()

                loss_val = total_loss.item()
                epoch_loss += loss_val * len(batch)
                num_samples += len(batch)
                pbar.set_postfix({
                    "loss": f"{loss_val:.4f}",
                    "recon": f"{recon_loss.item():.4f}",
                    "lr": f'{opt.param_groups[0]["lr"]:.2e}',
                })

        if num_samples == 0:
            raise RuntimeError("Training data yielded no batches")

        avg_loss = epoch_loss / num_samples
        scheduler.step(avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return best_state


def _compress(model: VQVAE, loader: DataLoader, out_path: Path, device: str) -> int:
    """Compress every corpus vector to PQ codes; returns the row count."""
    device = resolve_device(device)
    model.to(device)
    model.eval()

    chunks = []
    for batch in tqdm(loader, desc="Compressing"):
        chunks.append(model.compress(batch.to(device)))
    if not chunks:
        raise RuntimeError("No data was compressed")

    codes = np.concatenate(chunks, axis=0)
    np.save(out_path, codes)
    return codes.shape[0]


def train_corpus(corpus_npy: str,
                 input_dim: int,
                 epochs: int = 10,
                 num_subquantizers: int = 8,
                 codes_per_subquantizer: int = 256,
                 batch_size: int = 8192,
                 device: str = "cuda",
                 output_dir: Optional[str] = None,
                 learning_rate: float = 1e-3,
                 hidden_dim: int = 1024,
                 commitment_beta: float = 0.25,
                 embedding_model: Optional[str] = None,
                 quantizer: str = "pq",
                 use_rotation: bool = False,
                 rank_weight: float = 0.0,
                 store_embeddings: bool = True,
                 latent_bits: Optional[int] = None) -> str:
    """Train a product-quantized VQ-VAE on a corpus of embeddings.

    Args:
        corpus_npy: Path to a .npy file of shape (N, input_dim).
        input_dim: Dimension of the input embeddings.
        epochs: Training epochs.
        num_subquantizers: PQ subspaces (M); each document is stored as M codes.
        codes_per_subquantizer: Codebook size per subspace (K); K <= 256 keeps
            storage at one byte per code.
        batch_size: Training batch size.
        device: 'cuda' or 'cpu' (falls back to CPU when CUDA is unavailable).
        output_dir: Where to write artifacts (default: current directory).
        learning_rate: Adam learning rate.
        hidden_dim: Encoder/decoder hidden width.
        commitment_beta: VQ commitment loss weight.
        embedding_model: Recorded in metadata so the index knows which
            sentence-transformers model produced the corpus.
        quantizer: 'pq' (product quantization) or 'rq' (residual
            quantization; usually more accurate at the same byte budget).
        use_rotation: Learn an OPQ-style orthogonal rotation before
            quantization.
        rank_weight: Weight of the order-preserving ranking loss (0
            disables it).
        store_embeddings: Also save the corpus as float16
            (embeddings.f16.npy) so searches can rerank against the
            originals.
        latent_bits: Deprecated pre-v3 parameter; mapped to an equivalent
            number of subquantizers with a warning.

    Returns:
        Path to the output directory containing vqvae.pt, codes.npy,
        and vqvae_meta.json.
    """
    if latent_bits is not None:
        num_subquantizers = max(2, latent_bits // 8)
        warnings.warn(
            f"latent_bits is deprecated; using num_subquantizers="
            f"{num_subquantizers} ({num_subquantizers * 8} bits/doc) instead",
            DeprecationWarning, stacklevel=2,
        )

    if not os.path.exists(corpus_npy):
        raise FileNotFoundError(f"Corpus file not found: {corpus_npy}")
    if input_dim <= 0:
        raise ValueError(f"input_dim must be positive, got {input_dim}")
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")

    output_dir = Path(output_dir) if output_dir else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = NPZStreamDataset(corpus_npy, batch_size, input_dim)
    dl = DataLoader(ds, batch_size=None)

    model = VQVAE(
        input_dim,
        hidden=hidden_dim,
        num_subquantizers=num_subquantizers,
        codes_per_subquantizer=codes_per_subquantizer,
        beta=commitment_beta,
        quantizer=quantizer,
        use_rotation=use_rotation,
        rank_weight=rank_weight,
    )
    bytes_per_doc = model.codes_dtype.itemsize * num_subquantizers
    print(
        f"Training VQ-VAE ({quantizer.upper()}): latent_dim={model.lat_dim}, "
        f"M={num_subquantizers}, K={codes_per_subquantizer} -> "
        f"{bytes_per_doc} bytes/doc (vs {input_dim * 4} bytes float32)"
    )

    _train_loop(model, dl, epochs, device, learning_rate)

    torch.save(model.state_dict(), output_dir / "vqvae.pt")
    n_docs = _compress(model, dl, output_dir / "codes.npy", device)

    if store_embeddings:
        corpus = np.load(corpus_npy, mmap_mode="r")
        np.save(output_dir / "embeddings.f16.npy",
                np.asarray(corpus, dtype=np.float16))

    from .embedding import DEFAULT_EMBEDDING_MODEL
    meta = {
        "format_version": 3,
        "embedding_model": embedding_model or DEFAULT_EMBEDDING_MODEL,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "commitment_beta": commitment_beta,
        "num_documents": n_docs,
        **model.config(),
    }
    with open(output_dir / "vqvae_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Training complete. Artifacts in {output_dir}: vqvae.pt, codes.npy, vqvae_meta.json")
    return str(output_dir)

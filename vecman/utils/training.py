"""VECMAN training utilities for VQ-VAE model."""

import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from ..models.vqvae import VQVAE

class NPZStreamDataset(IterableDataset):
    def __init__(self, np_file: str, batch_size: int, input_dim: int):
        super().__init__()
        self.path = Path(np_file)
        self.bs = batch_size
        self._file = None
        self.d = input_dim
        
        # Validate file exists
        if not self.path.exists():
            raise FileNotFoundError(f"Corpus file not found: {self.path}")
        
        # Validate the numpy file can be loaded
        try:
            test_load = np.load(self.path, mmap_mode="r")
            if test_load.shape[1] != input_dim:
                raise ValueError(
                    f"Corpus dimension {test_load.shape[1]} doesn't match "
                    f"expected input_dim {input_dim}"
                )
        except Exception as e:
            raise ValueError(f"Invalid corpus file {self.path}: {e}")
    
    def __iter__(self):
        if self._file is None:
            self._file = np.load(self.path, mmap_mode="r")
        idx, N = 0, self._file.shape[0]
        while idx < N:
            batch = self._file[idx:idx + self.bs]
            # Ensure batch is float32 and handle any NaN values
            batch = np.nan_to_num(batch, nan=0.0, posinf=1.0, neginf=-1.0)
            yield torch.from_numpy(batch).float()
            idx += self.bs

def _train_loop(model: VQVAE, loader: DataLoader, epochs: int, device: str, learning_rate: float = 3e-4):
    """Internal training loop for VQ-VAE model."""
    # Validate device
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA requested but not available, falling back to CPU")
        device = "cpu"
    
    print(f"🚀 Training on device: {device}")
    model.to(device)
    model.train()
    
    opt = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.8, patience=2, verbose=True
    )
    
    best_loss = float('inf')
    
    for ep in range(1, epochs + 1):
        epoch_loss = 0.0
        num_batches = 0
        
        with tqdm(loader, desc=f"Epoch {ep}/{epochs}") as pbar:
            for batch in pbar:
                try:
                    batch = batch.to(device)
                    
                    # Forward pass
                    _, _, total_loss, recon_loss = model(batch)
                    
                    # Backward pass
                    opt.zero_grad()
                    total_loss.backward()
                    
                    # Gradient clipping to prevent explosion
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    opt.step()
                    
                    # Track metrics
                    loss_val = total_loss.item()
                    recon_val = recon_loss.item()
                    epoch_loss += loss_val * len(batch)
                    num_batches += len(batch)
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': f'{loss_val:.4f}',
                        'recon': f'{recon_val:.4f}',
                        'lr': f'{opt.param_groups[0]["lr"]:.2e}'
                    })
                    
                except Exception as e:
                    print(f"⚠️ Error in batch: {e}")
                    continue
        
        # Calculate average epoch loss
        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
            print(f"Epoch {ep}/{epochs} - avg_loss: {avg_loss:.4f}")
            
            # Update learning rate scheduler
            scheduler.step(avg_loss)
            
            # Track best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                print(f"🎯 New best loss: {best_loss:.4f}")
        else:
            print(f"⚠️ No valid batches processed in epoch {ep}")

def _compress(model: VQVAE, loader: DataLoader, out_path: str, device: str):
    """Compress the dataset using trained VQ-VAE model."""
    print(f"🗜️ Compressing dataset...")
    
    # Validate device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    
    model.to(device)
    model.eval()
    out = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Compressing"):
            try:
                batch = batch.to(device)
                _, idx, _, _ = model(batch)
                out.append(idx.cpu().numpy().astype(np.uint16))
            except Exception as e:
                print(f"⚠️ Error compressing batch: {e}")
                continue
    
    if out:
        compressed = np.concatenate(out)
        compressed.tofile(out_path)
        print(f"✅ Compressed {len(compressed)} vectors to {out_path}")
    else:
        raise RuntimeError("No data was successfully compressed")

def train_corpus(corpus_npy: str,
                input_dim: int,
                epochs: int = 10,  # Increased default epochs
                latent_bits: int = 16,
                batch_size: int = 8192,  # Increased from 4096 to 8192
                device: str = "cuda",
                output_dir: Optional[str] = None,
                learning_rate: float = 1e-3,  # Increased from 3e-4 to 1e-3
                hidden_dim: int = 1024,
                commitment_beta: float = 0.1) -> str:  # Added commitment loss weight
    """Train VQ-VAE model on a corpus of embeddings.
    
    Args:
        corpus_npy: Path to .npy file containing embeddings
        input_dim: Dimension of input embeddings
        epochs: Number of training epochs (increased default to 10)
        latent_bits: Number of bits for latent representation
        batch_size: Training batch size (increased default to 8192)
        device: Device to train on ('cuda' or 'cpu')
        output_dir: Directory to save model artifacts (default: current directory)
        learning_rate: Learning rate for optimizer (increased default to 1e-3)
        hidden_dim: Hidden layer dimension
        commitment_beta: Commitment loss weight for VQ-VAE (lower = less quantization pressure)
        
    Returns:
        Path to the output directory containing trained model
    """
    # Validate inputs
    if not os.path.exists(corpus_npy):
        raise FileNotFoundError(f"Corpus file not found: {corpus_npy}")
    
    if input_dim <= 0:
        raise ValueError(f"input_dim must be positive, got {input_dim}")
    
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")
    
    # Setup output directory
    output_dir = Path(output_dir) if output_dir else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🏋️ Training VQ-VAE:")
    print(f"  📁 Corpus: {corpus_npy}")
    print(f"  📊 Input dim: {input_dim}")
    print(f"  🔧 Latent bits: {latent_bits}")
    print(f"  ⚡ Device: {device}")
    print(f"  💾 Output: {output_dir}")
    
    try:
        # Create dataset and dataloader
        ds = NPZStreamDataset(corpus_npy, batch_size, input_dim)
        dl = DataLoader(ds, batch_size=None)
        
        # Create model
        model = VQVAE(input_dim, hidden=hidden_dim, latent_bits=latent_bits)
        # Update VQ layer commitment loss weight for better quantization  
        model.vq.beta = commitment_beta
        print(f"📐 Model created with latent dim: {model.lat_dim}, commitment beta: {commitment_beta}")
        
        # Train model
        _train_loop(model, dl, epochs, device, learning_rate)
        
        # Save model state
        model_path = output_dir / "vqvae.pt"
        torch.save(model.state_dict(), model_path)
        print(f"💾 Model saved: {model_path}")
        
        # Compress dataset
        codes_path = output_dir / "corpus.codes.bin"
        _compress(model, dl, codes_path, device)
        
        # Save metadata
        meta = {
            "input_dim": input_dim,
            "latent_bits": latent_bits,
            "latent_dim": model.lat_dim,
            "hidden_dim": hidden_dim,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "commitment_beta": commitment_beta
        }
        meta_path = output_dir / "vqvae_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        
        print("✅ Training completed successfully!")
        print(f"📁 Artifacts saved in: {output_dir}")
        print(f"   - {model_path.name}")
        print(f"   - {codes_path.name}")
        print(f"   - {meta_path.name}")
        
        return str(output_dir)
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise 
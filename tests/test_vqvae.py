"""Unit tests for the product-quantized VQ-VAE."""

import numpy as np
import pytest
import torch

from vecman.models.vqvae import (
    EMAVectorQuantizer,
    ProductQuantizer,
    VQVAE,
    codes_dtype,
    default_latent_dim,
)

DIM = 32


def test_codes_dtype_selection():
    assert codes_dtype(256) == np.dtype(np.uint8)
    assert codes_dtype(257) == np.dtype(np.uint16)
    assert codes_dtype(2 ** 16) == np.dtype(np.uint16)
    assert codes_dtype(2 ** 16 + 1) == np.dtype(np.uint32)


def test_default_latent_dim_divisible():
    for m in (2, 4, 8, 12):
        lat = default_latent_dim(384, m)
        assert lat % m == 0
        assert lat >= 64


def test_quantizer_straight_through_gradients():
    quantizer = EMAVectorQuantizer(num_codes=16, code_dim=8)
    quantizer.train()
    z = torch.randn(4, 8, requires_grad=True)
    z_q, idx, loss = quantizer(z)
    (z_q.sum() + loss).backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert idx.shape == (4,)
    assert idx.max() < 16


def test_quantizer_output_is_codebook_entry_in_eval():
    quantizer = EMAVectorQuantizer(num_codes=16, code_dim=8)
    quantizer.eval()
    z = torch.randn(4, 8)
    z_q, idx, _ = quantizer(z)
    # With the straight-through trick, forward value equals the codebook entry.
    assert torch.allclose(z_q, quantizer.codebook[idx], atol=1e-6)


def test_product_quantizer_roundtrip_shapes():
    pq = ProductQuantizer(code_dim=32, num_subquantizers=4, codes_per_subquantizer=16)
    pq.eval()
    z = torch.randn(10, 32)
    z_q, idx, loss = pq(z)
    assert z_q.shape == (10, 32)
    assert idx.shape == (10, 4)
    codes = pq.encode_indices(z)
    assert torch.equal(codes, idx)
    decoded = pq.decode_indices(codes)
    assert torch.allclose(decoded, z_q, atol=1e-6)


def test_product_quantizer_rejects_indivisible_dim():
    with pytest.raises(ValueError):
        ProductQuantizer(code_dim=30, num_subquantizers=4)


def test_vqvae_forward_and_compress():
    model = VQVAE(DIM, hidden=64, latent_dim=16, num_subquantizers=4,
                  codes_per_subquantizer=16)
    model.eval()
    x = torch.randn(6, DIM)
    recon, idx, total_loss, recon_loss = model(x)
    assert recon.shape == x.shape
    assert idx.shape == (6, 4)
    assert total_loss.item() >= recon_loss.item() - 1e-6

    codes = model.compress(x)
    assert codes.shape == (6, 4)
    assert codes.dtype == np.uint8

    latents = model.decompress(codes)
    assert latents.shape == (6, 16)


def test_vqvae_rejects_bad_input_dim():
    model = VQVAE(DIM, hidden=64, latent_dim=16, num_subquantizers=4)
    with pytest.raises(ValueError):
        model(torch.randn(2, DIM + 1))
    with pytest.raises(ValueError):
        VQVAE(-1)


def test_vqvae_config_roundtrip():
    model = VQVAE(DIM, hidden=64, latent_dim=16, num_subquantizers=4,
                  codes_per_subquantizer=32)
    clone = VQVAE.from_config(model.config())
    clone.load_state_dict(model.state_dict())
    x = torch.randn(3, DIM)
    model.eval(); clone.eval()
    assert np.array_equal(model.compress(x), clone.compress(x))


def test_ema_training_moves_codebook():
    quantizer = EMAVectorQuantizer(num_codes=8, code_dim=4)
    quantizer.train()
    before = quantizer.codebook.clone()
    for _ in range(5):
        quantizer(torch.randn(64, 4) + 3.0)
    assert not torch.allclose(before, quantizer.codebook)

"""End-to-end training pipeline tests (no network / model downloads)."""

import json

import numpy as np
import pytest
import torch

from vecman.core.index import VecmanIndex
from vecman.models.vqvae import VQVAE
from vecman.utils.retrieval import load_assets, retrieve, save_jsonl
from vecman.utils.training import NPZStreamDataset, train_corpus

DIM = 32
N = 64


@pytest.fixture
def corpus(tmp_path):
    np.random.seed(1)
    vectors = np.random.randn(N, DIM).astype(np.float32)
    path = tmp_path / "corpus.npy"
    np.save(path, vectors)
    return path, vectors


def test_stream_dataset_batches(corpus):
    path, vectors = corpus
    ds = NPZStreamDataset(str(path), batch_size=20, input_dim=DIM)
    batches = list(ds)
    assert sum(len(b) for b in batches) == N
    assert all(b.dtype == torch.float32 for b in batches)


def test_stream_dataset_validates(tmp_path, corpus):
    path, _ = corpus
    with pytest.raises(FileNotFoundError):
        NPZStreamDataset(str(tmp_path / "missing.npy"), 8, DIM)
    with pytest.raises(ValueError):
        NPZStreamDataset(str(path), 8, DIM + 5)


def test_train_corpus_artifacts(tmp_path, corpus):
    path, vectors = corpus
    out = train_corpus(
        str(path), input_dim=DIM, epochs=2, num_subquantizers=4,
        codes_per_subquantizer=16, batch_size=32, device="cpu",
        output_dir=str(tmp_path / "model"), hidden_dim=64,
    )
    out_files = {p.name for p in (tmp_path / "model").iterdir()}
    assert {"vqvae.pt", "codes.npy", "vqvae_meta.json"} <= out_files

    codes = np.load(tmp_path / "model" / "codes.npy")
    assert codes.shape == (N, 4)
    assert codes.dtype == np.uint8  # 16 codes per subquantizer fits in uint8

    with open(tmp_path / "model" / "vqvae_meta.json") as f:
        meta = json.load(f)
    assert meta["format_version"] == 3
    assert meta["num_documents"] == N

    # The trained model must load and reproduce the stored codes.
    model = VQVAE.from_config(meta)
    model.load_state_dict(torch.load(tmp_path / "model" / "vqvae.pt",
                                     map_location="cpu"))
    model.eval()
    recomputed = model.compress(torch.from_numpy(vectors))
    assert np.array_equal(recomputed, codes)


def test_latent_bits_deprecation(tmp_path, corpus):
    path, _ = corpus
    with pytest.warns(DeprecationWarning):
        train_corpus(
            str(path), input_dim=DIM, epochs=1, latent_bits=16,
            batch_size=32, device="cpu", output_dir=str(tmp_path / "m2"),
            hidden_dim=64,
        )
    codes = np.load(tmp_path / "m2" / "codes.npy")
    assert codes.shape[1] == 2  # 16 bits -> 2 subquantizers


def test_legacy_retrieve_uses_stored_codes(tmp_path, corpus):
    path, vectors = corpus
    out = train_corpus(
        str(path), input_dim=DIM, epochs=2, num_subquantizers=4,
        codes_per_subquantizer=16, batch_size=32, device="cpu",
        output_dir=str(tmp_path / "model"), hidden_dim=64,
    )
    docs = [f"doc-{i}" for i in range(N)]
    save_jsonl(docs, str(tmp_path / "model" / "docs.jsonl"))

    vqvae, codes, loaded_docs = load_assets(out)
    assert loaded_docs == docs
    assert codes.ndim == 2

    result_docs, scores = retrieve(vqvae, codes, docs, vectors[0], k=5,
                                   method="vqvae")
    assert len(result_docs) == 5
    assert len(scores) == 5
    assert all(-1.0 <= s <= 1.0 for s in scores)


def test_retrieve_rejects_legacy_flat_codes():
    model = VQVAE(DIM, hidden=64, latent_dim=16, num_subquantizers=4)
    flat = np.zeros(10, dtype=np.uint16)
    with pytest.raises(ValueError, match="pre-v3"):
        retrieve(model, flat, ["d"] * 10, np.random.randn(DIM).astype(np.float32))


def test_train_corpus_input_validation(tmp_path, corpus):
    path, _ = corpus
    with pytest.raises(FileNotFoundError):
        train_corpus("missing.npy", input_dim=DIM, device="cpu")
    with pytest.raises(ValueError):
        train_corpus(str(path), input_dim=0, device="cpu")
    with pytest.raises(ValueError):
        train_corpus(str(path), input_dim=DIM, epochs=0, device="cpu")


def test_index_load_from_training_dir(tmp_path, corpus):
    """VecmanIndex.load works directly on a train_corpus output dir + docs."""
    path, vectors = corpus
    out = train_corpus(
        str(path), input_dim=DIM, epochs=2, num_subquantizers=4,
        codes_per_subquantizer=16, batch_size=32, device="cpu",
        output_dir=str(tmp_path / "model"), hidden_dim=64,
    )
    save_jsonl([f"doc-{i}" for i in range(N)],
               str(tmp_path / "model" / "docs.jsonl"))
    index = VecmanIndex.load(out)
    assert len(index) == N
    results = index.search(vectors[0], k=3)
    assert len(results) == 3

"""Tests for VecmanIndex: CRUD, search, filtering, persistence."""

import numpy as np
import pytest
import torch

from vecman.core.index import VecmanIndex, _kmeans
from vecman.models.vqvae import VQVAE

DIM = 32
N_DOCS = 40


@pytest.fixture
def trained_index():
    """A small trained model + index over clustered synthetic vectors."""
    torch.manual_seed(0)
    np.random.seed(0)
    model = VQVAE(DIM, hidden=64, latent_dim=16, num_subquantizers=4,
                  codes_per_subquantizer=16)

    # Two well-separated clusters so nearest-neighbour structure survives
    # compression even with an undertrained model.
    center_a = np.random.randn(DIM).astype(np.float32) * 5
    center_b = -center_a
    vectors = np.stack([
        (center_a if i < N_DOCS // 2 else center_b) +
        np.random.randn(DIM).astype(np.float32) * 0.1
        for i in range(N_DOCS)
    ])

    # Brief training so codebooks adapt to the data.
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    data = torch.from_numpy(vectors)
    for _ in range(30):
        _, _, loss, _ = model(data)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()

    index = VecmanIndex(model)
    texts = [f"doc-{i}" for i in range(N_DOCS)]
    metas = [{"cluster": "a" if i < N_DOCS // 2 else "b"} for i in range(N_DOCS)]
    index.add_vectors(vectors, texts, metas)
    return index, vectors


def test_add_and_len(trained_index):
    index, _ = trained_index
    assert len(index) == N_DOCS
    assert index.codes.shape == (N_DOCS, 4)
    assert index.codes.dtype == np.uint8


def test_search_returns_same_cluster(trained_index):
    index, vectors = trained_index
    results = index.search(vectors[0], k=5)
    assert len(results) == 5
    # Scores are sorted descending and are valid cosines.
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert all(-1.0 <= s <= 1.0 for s in scores)
    # All results should come from the query's own cluster.
    assert all(r.metadata["cluster"] == "a" for r in results)


def test_metadata_filter(trained_index):
    index, vectors = trained_index
    results = index.search(vectors[0], k=5, filter={"cluster": "b"})
    assert len(results) == 5
    assert all(r.metadata["cluster"] == "b" for r in results)


def test_filter_no_match(trained_index):
    index, vectors = trained_index
    assert index.search(vectors[0], k=5, filter={"cluster": "zzz"}) == []


def test_delete(trained_index):
    index, vectors = trained_index
    target = index.search(vectors[0], k=1)[0]
    assert index.delete(target.id) == 1
    assert len(index) == N_DOCS - 1
    ids = [r.id for r in index.search(vectors[0], k=N_DOCS - 1)]
    assert target.id not in ids
    # Deleting again is a no-op.
    assert index.delete(target.id) == 0


def test_update_metadata_and_vector(trained_index):
    index, vectors = trained_index
    index.update(0, metadata={"cluster": "updated"})
    assert index.metadata[0] == {"cluster": "updated"}
    # Move doc 0 onto cluster b's side: it must compress to the same codes
    # as the doc whose vector it copied, and score as high as the top hit.
    index.update(0, vector=vectors[-1])
    assert np.array_equal(index.codes[0], index.codes[N_DOCS - 1])
    results = index.search(vectors[-1], k=N_DOCS)
    score_by_id = {r.id: r.score for r in results}
    assert score_by_id[0] == pytest.approx(max(score_by_id.values()))
    with pytest.raises(KeyError):
        index.update(999)


def test_incremental_add(trained_index):
    index, vectors = trained_index
    new_ids = index.add_vectors(vectors[:2] * 1.01, ["new-1", "new-2"])
    assert new_ids == [N_DOCS, N_DOCS + 1]
    assert len(index) == N_DOCS + 2


def test_save_load_roundtrip(tmp_path, trained_index):
    index, vectors = trained_index
    index.delete(3)
    index.save(tmp_path)

    loaded = VecmanIndex.load(tmp_path)
    assert len(loaded) == len(index)
    assert loaded.docs == index.docs
    assert np.array_equal(loaded.codes, index.codes)
    assert np.array_equal(loaded.alive, index.alive)

    original = [r.id for r in index.search(vectors[0], k=5)]
    reloaded = [r.id for r in loaded.search(vectors[0], k=5)]
    assert original == reloaded


def test_load_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        VecmanIndex.load(tmp_path / "nope")


def test_dimension_mismatch_raises(trained_index):
    index, _ = trained_index
    with pytest.raises(ValueError):
        index.add_vectors(np.random.randn(2, DIM + 1).astype(np.float32),
                          ["x", "y"])


def test_kmeans_shapes():
    x = np.random.randn(100, 8).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    centroids = _kmeans(x, 5)
    assert centroids.shape == (5, 8)
    assert np.isfinite(centroids).all()

"""Tests for v3.1 features: RQ, rotation, ADC, rerank, hybrid, HNSW,
vector-space selection ops, rich filters, compact."""

import numpy as np
import pytest
import torch

from vecman.core.bm25 import BM25
from vecman.core.hnsw import HNSW
from vecman.core.index import VecmanIndex
from vecman.models.vqvae import VQVAE, ResidualQuantizer

DIM = 32
N_DOCS = 60


def _train_model(vectors: np.ndarray, **model_kwargs) -> VQVAE:
    defaults = dict(hidden=64, latent_dim=16, num_subquantizers=4,
                    codes_per_subquantizer=16)
    defaults.update(model_kwargs)
    model = VQVAE(DIM, **defaults)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    data = torch.from_numpy(vectors)
    for _ in range(40):
        _, _, loss, _ = model(data)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return model


@pytest.fixture(scope="module")
def clustered_vectors():
    rng = np.random.default_rng(7)
    centers = rng.normal(size=(6, DIM)).astype(np.float32) * 4
    vectors = np.stack([
        centers[i % 6] + rng.normal(size=DIM).astype(np.float32) * 0.3
        for i in range(N_DOCS)
    ])
    return vectors


@pytest.fixture(scope="module")
def index(clustered_vectors):
    model = _train_model(clustered_vectors)
    idx = VecmanIndex(model)
    texts = [f"doc-{i} cluster-{i % 6}" for i in range(N_DOCS)]
    metas = [{"cluster": i % 6, "tag": "even" if i % 2 == 0 else "odd"}
             for i in range(N_DOCS)]
    idx.add_vectors(clustered_vectors, texts, metas)
    return idx


# ------------------------------------------------------------ quantizers

def test_residual_quantizer_roundtrip():
    rq = ResidualQuantizer(code_dim=16, num_stages=4, codes_per_stage=8)
    rq.eval()
    z = torch.randn(10, 16)
    z_q, idx, loss = rq(z)
    assert z_q.shape == (10, 16)
    assert idx.shape == (10, 4)
    codes = rq.encode_indices(z)
    assert torch.equal(codes, idx)
    decoded = rq.decode_indices(codes)
    assert torch.allclose(decoded, z_q, atol=1e-5)


def test_rq_reconstructs_better_than_single_stage():
    torch.manual_seed(0)
    z = torch.randn(200, 16)
    rq4 = ResidualQuantizer(16, num_stages=4, codes_per_stage=32)
    rq1 = ResidualQuantizer(16, num_stages=1, codes_per_stage=32)
    for rq in (rq4, rq1):
        rq.train()
        for _ in range(50):
            rq(z)
        rq.eval()
    err4 = (rq4.decode_indices(rq4.encode_indices(z)) - z).pow(2).mean()
    err1 = (rq1.decode_indices(rq1.encode_indices(z)) - z).pow(2).mean()
    assert err4 < err1


def test_vqvae_rq_mode(clustered_vectors):
    model = _train_model(clustered_vectors, quantizer="rq")
    codes = model.compress(torch.from_numpy(clustered_vectors))
    assert codes.shape == (N_DOCS, 4)
    latents = model.decompress(codes)
    assert latents.shape == (N_DOCS, 16)
    clone = VQVAE.from_config(model.config())
    assert clone.quantizer_type == "rq"
    clone.load_state_dict(model.state_dict())


def test_rotation_is_orthogonal(clustered_vectors):
    model = _train_model(clustered_vectors, use_rotation=True)
    weight = model.rotation.weight.detach()
    identity = torch.eye(weight.shape[0])
    assert torch.allclose(weight @ weight.t(), identity, atol=1e-4)
    clone = VQVAE.from_config(model.config())
    clone.load_state_dict(model.state_dict())
    x = torch.randn(3, DIM)
    assert np.array_equal(model.compress(x), clone.compress(x))


def test_ranking_loss_finite():
    x = torch.randn(16, DIM)
    z = torch.randn(16, 16, requires_grad=True)
    loss = VQVAE._ranking_loss(x, z)
    assert torch.isfinite(loss)
    if loss.requires_grad:
        loss.backward()


# ---------------------------------------------------------------- search

def test_adc_matches_latent_scores(index, clustered_vectors):
    q = clustered_vectors[0]
    latent_results = index.search(q, k=10, method="latent")
    adc_results = index.search(q, k=10, method="adc")
    assert [r.id for r in latent_results] == [r.id for r in adc_results]
    for a, b in zip(latent_results, adc_results):
        assert a.score == pytest.approx(b.score, abs=1e-4)


def test_rerank_uses_exact_similarity(index, clustered_vectors):
    q = clustered_vectors[3]
    reranked = index.search(q, k=5, rerank=True)
    assert len(reranked) == 5
    # The query is doc 3 itself; exact rerank must put it first with cos ~1.
    assert reranked[0].id == 3
    assert reranked[0].score == pytest.approx(1.0, abs=1e-3)


def test_rerank_without_embeddings_raises(index, clustered_vectors):
    model = index.model
    bare = VecmanIndex(model, store_embeddings=False)
    bare.add_vectors(clustered_vectors[:5], [f"d{i}" for i in range(5)])
    with pytest.raises(RuntimeError, match="store_embeddings"):
        bare.search(clustered_vectors[0], k=2, rerank=True)


def test_search_batch(index, clustered_vectors):
    batches = index.search_batch([clustered_vectors[0],
                                  clustered_vectors[1]], k=3)
    assert len(batches) == 2
    assert all(len(b) == 3 for b in batches)
    single = index.search(clustered_vectors[0], k=3)
    assert [r.id for r in batches[0]] == [r.id for r in single]


def test_find_similar(index):
    results = index.find_similar(0, k=5)
    assert len(results) == 5
    assert all(r.id != 0 for r in results)
    # Same-cluster docs should dominate the neighbours of doc 0.
    same_cluster = sum(1 for r in results if r.metadata["cluster"] == 0)
    assert same_cluster >= 3


def test_range_search(index, clustered_vectors):
    results = index.range_search(clustered_vectors[0], min_score=0.5)
    assert len(results) >= 1
    assert all(r.score >= 0.5 for r in results)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_hybrid_search_finds_keyword(index, clustered_vectors):
    # 'cluster-4' appears verbatim in every cluster-4 doc text; the dense
    # query is a vector, so the keyword side is passed explicitly.
    results = index.search(clustered_vectors[4], k=5, hybrid=True,
                           keyword_query="cluster-4")
    assert len(results) == 5
    hits = sum(1 for r in results if "cluster-4" in r.text)
    assert hits >= 3


def test_hybrid_vector_query_requires_keyword(index, clustered_vectors):
    with pytest.raises(ValueError, match="keyword_query"):
        index.search(clustered_vectors[0], k=3, hybrid=True)


# ---------------------------------------------------------------- filters

def test_filter_operators(index, clustered_vectors):
    q = clustered_vectors[0]
    assert all(r.metadata["cluster"] in (0, 1)
               for r in index.search(q, k=10, filter={"cluster": {"$in": [0, 1]}}))
    assert all(r.metadata["cluster"] > 2
               for r in index.search(q, k=10, filter={"cluster": {"$gt": 2}}))
    assert all(r.metadata["cluster"] <= 1
               for r in index.search(q, k=10, filter={"cluster": {"$lte": 1}}))
    assert all(r.metadata["tag"] != "odd"
               for r in index.search(q, k=10, filter={"tag": {"$ne": "odd"}}))
    assert all("cluster-2" in r.text
               for r in index.search(q, k=10,
                                     filter={"tag": {"$in": ["even", "odd"]},
                                             "cluster": 2}))


def test_filter_contains(index, clustered_vectors):
    results = index.search(clustered_vectors[0], k=5,
                           filter={"tag": {"$contains": "ev"}})
    assert all(r.metadata["tag"] == "even" for r in results)


def test_filter_unknown_operator(index, clustered_vectors):
    with pytest.raises(ValueError, match="Unknown filter operator"):
        index.search(clustered_vectors[0], k=3, filter={"cluster": {"$foo": 1}})


# ----------------------------------------------------------- maintenance

def test_compact(clustered_vectors):
    model = _train_model(clustered_vectors)
    idx = VecmanIndex(model)
    idx.add_vectors(clustered_vectors[:10], [f"d{i}" for i in range(10)])
    idx.delete([2, 5])
    mapping = idx.compact()
    assert len(idx) == 8
    assert len(idx.docs) == 8
    assert 2 not in mapping and 5 not in mapping
    assert mapping[3] == 2  # shifted down past deleted id 2
    results = idx.search(clustered_vectors[0], k=8)
    assert len(results) == 8


def test_save_load_with_embeddings(tmp_path, index, clustered_vectors):
    index.save(tmp_path)
    assert (tmp_path / "embeddings.f16.npy").exists()
    loaded = VecmanIndex.load(tmp_path)
    assert loaded._embeddings is not None
    reranked = loaded.search(clustered_vectors[3], k=3, rerank=True)
    assert reranked[0].id == 3


# ----------------------------------------------------------------- hnsw

def test_hnsw_recall_on_clusters():
    rng = np.random.default_rng(3)
    centers = rng.normal(size=(10, 16)).astype(np.float32)
    vectors = np.concatenate([
        centers[i] + rng.normal(size=(30, 16)).astype(np.float32) * 0.05
        for i in range(10)
    ])
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    graph = HNSW(dim=16, m=8, ef_construction=100)
    for row in vectors:
        graph.add(row)
    vectors_t = torch.from_numpy(vectors)  # torch matmul: see _matmul note
    hits = 0
    for probe in range(0, 300, 30):
        exact = int(torch.argmax(vectors_t @ vectors_t[probe]))
        found = graph.search(vectors[probe], k=1)
        if found and found[0][1] == exact:
            hits += 1
    assert hits >= 8  # >= 80% top-1 recall on easy clustered data


def test_hnsw_ann_path(index, clustered_vectors):
    results = index.search(clustered_vectors[0], k=5, ann="hnsw")
    assert len(results) == 5
    flat = index.search(clustered_vectors[0], k=5, ann="flat")
    # On this tiny corpus HNSW should agree with the exhaustive scan.
    assert set(r.id for r in results) & set(r.id for r in flat)


# ----------------------------------------------------------------- bm25

def test_bm25_ranks_matching_doc_first():
    docs = ["the cat sat on the mat", "dogs chase cats sometimes",
            "quantum computing is hard", "my cat naps all day"]
    bm25 = BM25(docs)
    scores = bm25.scores("cat")
    assert scores[0] > 0 and scores[3] > 0
    assert scores[2] == 0.0


def test_bm25_empty_query():
    bm25 = BM25(["a b c"])
    assert bm25.scores("").sum() == 0

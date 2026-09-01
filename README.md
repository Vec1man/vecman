<div align="center">

# VECMAN

![VECMAN Logo](media/VV.png)

**Learned embedding compression + compressed vector search.**
Store each document as **8 bytes** instead of 1,536 — and search it without decompressing.

[![CI](https://github.com/Vec1man/vecman/actions/workflows/ci.yml/badge.svg)](https://github.com/Vec1man/vecman/actions)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://github.com/Vec1man/vecman)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Version](https://img.shields.io/badge/version-3.1.0-orange)](CHANGELOG.md)

[Quick start](#-quick-start) · [Examples](#-examples) · [Benchmarks](#-benchmarks) · [vs other solutions](#-how-vecman-compares-to-other-solutions) · [API docs](#-api-reference) · [Architecture](#-architecture)

</div>

---

## What is VECMAN?

VECMAN trains a **product- or residual-quantized VQ-VAE** on your text embeddings and stores every document as a handful of discrete codes. Search runs **directly in the compressed latent space** — the corpus is never re-embedded and (with ADC) never even decompressed per query. An optional second stage reranks the top candidates against float16 originals for near-exact quality.

```
        1,536 bytes                                 8 bytes
┌──────────────────────────┐              ┌──────────────────────┐
│ float32 embedding (384d) │──► encoder ──│ [ 17, 203, 4, 88,    │
│  0.0231, -0.1187, ...    │   + learned  │   251, 9, 130, 66 ]  │
└──────────────────────────┘   codebooks  └──────────────────────┘
                                              192× smaller
```

**It is not a hosted database.** It's a compression + retrieval layer that runs anywhere PyTorch runs — embed it in your app, serve it over the built-in REST API, or plug it into LangChain.

### Feature highlights

| | |
|---|---|
| 🗜️ **Learned compression** | PQ or residual quantization (RQ), trained end-to-end with EMA codebooks, dead-code reinit, and an anti-collapse similarity loss |
| ⚡ **ADC search** | Query → one small `(M×K)` lookup table → M gathers per doc. RAM stays proportional to the codes |
| 🎯 **Two-stage rerank** | recall@10 **0.83 → 0.98** at 64× compression, +0.04 ms/query (measured, see below) |
| 🕸️ **ANN structures** | IVF (spherical k-means++) and a dependency-free HNSW graph |
| 🔎 **Hybrid search** | BM25 keyword scores fused with dense scores via reciprocal-rank fusion |
| 🏷️ **Rich filters** | `$in`, `$nin`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$contains` |
| 🔁 **Full CRUD** | Incremental add (no retrain), delete, update, `compact()` |
| 🖥️ **GPU** | `index.to("cuda")` moves all search math to the GPU |
| 🌐 **Zero-dep REST API** | `vecman serve` — stdlib only |
| 🦜 **LangChain adapter** | `VecmanVectorStore` drop-in |

---

## 📦 Installation

```bash
pip install vecman            # core: compression + search
pip install vecman[rag]       # + Gemini answer generation
pip install vecman[eval]      # + RAGAS evaluation
pip install vecman[dev]       # + pytest (contributors)
```

Requires Python ≥ 3.9 and PyTorch ≥ 2.0.

---

## 🚀 Quick start

```python
import numpy as np
from vecman import VecmanIndex, embed_texts, save_jsonl, train_corpus

texts = ["Machine learning is...", "Deep learning uses...", "..."]

# 1. Embed and train (one-time)
embeddings = embed_texts(texts)                       # sentence-transformers, cached model
np.save("index/corpus.npy", embeddings)
save_jsonl(texts, "index/docs.jsonl")
train_corpus("index/corpus.npy", input_dim=embeddings.shape[1],
             epochs=10, device="cpu", output_dir="index")

# 2. Load and search — codes are decompressed once, then every query is
#    a single encoder pass + one matmul (or a tiny lookup table with ADC)
index = VecmanIndex.load("index")
for r in index.search("What is machine learning?", k=5):
    print(f"[{r.score:.3f}] {r.text}")
```

Or entirely from the command line:

```bash
vecman index docs.txt -o my_index --epochs 10 --quantizer rq
vecman query "what is machine learning" -d my_index -k 5 --rerank
vecman info -d my_index
vecman serve -d my_index -p 8080
```

---

## 📚 Examples

### CRUD — no retraining needed

```python
ids = index.add_texts(
    ["A new document about optimization"],
    metadatas=[{"lang": "en", "year": 2026}],
)                                  # compressed through the trained encoder
index.update(ids[0], metadata={"lang": "en", "year": 2025, "reviewed": True})
index.delete(3)                    # soft delete
index.compact()                    # physically drop deleted rows, returns {old_id: new_id}
index.save("index")
```

### Two-stage reranking (recommended default)

Compressed codes find the candidates; float16 originals settle the order.
Near-exact quality at compressed-index memory cost:

```python
index.search("query", k=5, rerank=True)              # recall@10 0.98 on our benchmark
index.search("query", k=5, rerank=True, rerank_multiplier=20)  # wider candidate pool
```

### Hybrid search (keywords + meaning)

Dense retrieval is weak on exact names, IDs and rare terms — BM25 isn't.
Fuse both with reciprocal-rank fusion:

```python
index.search("error code E4402", k=5, hybrid=True)
# with a vector query, pass the keyword text separately:
index.search(query_vector, k=5, hybrid=True, keyword_query="E4402")
```

### Metadata filtering

```python
index.search("query", k=5, filter={"lang": "en"})                    # equality
index.search("query", k=5, filter={"year": {"$gte": 2024}})          # range
index.search("query", k=5, filter={"lang": {"$in": ["en", "ar"]},
                                   "title": {"$contains": "intro"},
                                   "draft": {"$ne": True}})
```

### Vector-space selection

```python
index.search_batch(["q1", "q2", "q3"], k=5)     # one batched embedding call
index.find_similar(doc_id=7, k=5)               # more-like-this, straight from stored codes
index.range_search("query", min_score=0.6)      # everything above a similarity threshold
```

### Scaling knobs

```python
index.search("query", k=5, method="adc")        # lookup-table scoring: lowest RAM
index.search("query", k=5, ann="ivf", nprobe=16)  # cluster-pruned scan (auto ≥ 10k docs)
index.search("query", k=5, ann="hnsw")          # graph ANN
index.to("cuda")                                 # GPU search math
```

### Higher-accuracy training

```python
train_corpus(
    "corpus.npy", input_dim=384, epochs=20,
    quantizer="rq",        # residual quantization: staged full-dim codebooks
    use_rotation=True,     # OPQ-style learned orthogonal rotation
    rank_weight=0.2,       # order-preserving triplet loss on top-k structure
    num_subquantizers=8,   # bytes per document (at K ≤ 256)
    device="cuda",
)
```

### RAG in three lines

```python
from vecman import generate_answer          # pip install vecman[rag] + GOOGLE_API_KEY

contexts = [r.text for r in index.search(question, k=5, rerank=True)]
answer = generate_answer(question, contexts)
```

### REST API (stdlib, zero dependencies)

```bash
vecman serve -d my_index -p 8080
```

```
GET  /search?q=...&k=5&rerank=1&hybrid=1     → {"results": [{id, text, score, metadata}]}
GET  /info      GET /health
POST /search    {"queries": ["q1","q2"], "k": 5, "filter": {...}, "rerank": true}
POST /add       {"texts": [...], "metadatas": [...]}
POST /delete    {"ids": [3, 7]}
POST /save      {}
```

### LangChain

```python
from vecman.integrations.langchain import VecmanVectorStore   # needs langchain-core

store = VecmanVectorStore(index, rerank=True)
store.similarity_search("query", k=5)
store.add_texts(["new doc"], metadatas=[{"src": "api"}])
```

---

## 📊 Benchmarks

Reproduce everything with the bundled script — **we only publish numbers that come out of it**:

```bash
python benchmarks/benchmark.py --n 2000 --dim 128 --epochs 40 --queries 50 \
       --k 10 --clusters 200 --batch-size 256              # PQ row
python benchmarks/benchmark.py ... --quantizer rq --rank-weight 0.2   # RQ row
```

Measured on CPU, synthetic clustered embeddings, v3.1.0 (recall is against exact float32 cosine search as ground truth):

| Metric | PQ | RQ (+rank loss) |
|--------|-----|-----|
| recall@10, compressed only | 0.83 | 0.84 |
| recall@10, ADC scoring | 0.83 *(identical)* | 0.84 *(identical)* |
| **recall@10 with rerank** | **0.98** | **0.99** |
| bytes/doc (codes) | 8 (vs 512 raw float32) | 8 |
| compression ratio | 64× | 64× |
| avg query, compressed | 0.7 ms | 2.8 ms |
| avg query, reranked | 0.7 ms | 3.0 ms |

Notes worth knowing before you benchmark your own data:

- **Recall depends on data geometry.** Distinguishing near-duplicate documents from 8-byte codes is fundamentally hard (raise `num_subquantizers` for finer resolution). Topical retrieval — the typical RAG case — holds up well.
- **ADC is score-identical to the latent path** (there's a test asserting it); it trades a bit of per-query CPU for not materializing the latent matrix.
- At 384-dim the raw/compressed ratio becomes **192×** for the codes; with rerank enabled you also keep float16 originals (768 bytes/doc), still **2×** smaller than raw plus the recall of near-exact search.

Community benchmark results (with the exact command used) are welcome as issues or PRs. A BEIR/MTEB harness is on the roadmap.

---

## ⚖️ How VECMAN compares to other solutions

Honest positioning — VECMAN is a **compression + retrieval layer**, not a managed distributed database. Pick by problem:

| | VECMAN | FAISS (PQ/OPQ) | Qdrant / Milvus | pgvector | Pinecone |
|---|---|---|---|---|---|
| Type | Python library + optional server | C++ library | Server / cluster | Postgres extension | Managed SaaS |
| Compression | **Learned** (neural PQ/RQ, trained end-to-end) | Classical PQ/OPQ (k-means) | Scalar/PQ quantization | None (raw vectors) | Internal |
| Codes trained on *your* data's semantics | ✅ encoder + codebooks + ranking loss | codebooks only | ❌ | ❌ | ❌ |
| Two-stage rerank built in | ✅ | manual | ✅ | manual | ✅ |
| Hybrid BM25 + dense | ✅ (RRF) | ❌ | ✅ | via `tsvector` | ✅ |
| Metadata filter operators | ✅ | ❌ (IDMap tricks) | ✅ (richer) | ✅ (SQL — richest) | ✅ |
| CRUD without rebuild | ✅ | partial | ✅ | ✅ | ✅ |
| Horizontal scaling / replication | ❌ | ❌ | ✅ | via Postgres | ✅ |
| Runs fully offline / embedded | ✅ | ✅ | self-host | self-host | ❌ |
| Dependencies | numpy, torch, sentence-transformers | faiss | server deploy | Postgres | account |

**Choose VECMAN when** embedding storage/RAM is the bottleneck and you want compression that *adapts to your corpus* (the encoder, codebooks, and ranking loss are all trained on your data — classical PQ only fits codebooks); or when you want a single `pip install` retrieval stack with hybrid search, rerank and a REST API, no server to operate.

**Choose FAISS when** you need battle-tested C++ speed at 100M+ scale and are happy with classical quantization.

**Choose Qdrant/Milvus/pgvector/Pinecone when** you need a database: replication, multi-tenant auth, backups, horizontal scale. (A VECMAN-style learned quantizer can still be the compression layer in front of them — that interop is on the roadmap.)

---

## 🏗️ Architecture

```
                       TRAIN (once)
  corpus.npy ──► Encoder MLP ──► BatchNorm ──► [rotation] ──► Quantizer ──► Decoder MLP
   (N × 384)      384→96          anti-collapse   (OPQ opt.)   PQ: M×(K×12)    96→384
                                                               RQ: L stages    recon loss
                     losses: reconstruction + commitment + similarity + ranking

                       INDEX
  codes.npy (N × M uint8)   docs.jsonl (text + metadata)   embeddings.f16.npy (rerank, opt.)

                       SEARCH (per query)
  query ──► encode (384→96, centered) ──► candidates ──► score ──► [rerank] ──► top-k
                                          flat | IVF | HNSW   latent matmul | ADC tables
```

Key design decisions, in code:

- **Anti-collapse stack** ([vqvae.py](vecman/models/vqvae.py)): a pre-quantizer BatchNorm plus a pairwise similarity-preservation loss. Without these the encoder satisfies reconstruction while mapping every document to nearly the same direction — retrieval silently dies. (Found the hard way; there's a before/after in the [v3.0.0 changelog](CHANGELOG.md).)
- **Centering at search time** ([index.py](vecman/core/index.py)): latents are centered on the corpus mean before cosine; a shared offset is harmless for reconstruction but pushes all cosines toward 1.
- **EMA codebooks + dead-code reinit**: unused codes are re-seeded from live batch latents every step.
- **HNSW heuristic neighbour selection** ([hnsw.py](vecman/core/hnsw.py)): pruning purely by similarity disconnects the graph on clustered data (top-1 recall 20%); Malkov & Yashunin's Algorithm 4 keeps long-range links (80%+).
- **All matmuls through torch**: mixing numpy-MKL and torch BLAS in one process crashes on some Anaconda setups (duplicate OpenMP runtimes).

### On-disk format (`format_version: 3`)

| File | Contents |
|---|---|
| `vqvae.pt` | model weights |
| `vqvae_meta.json` | architecture + embedding model name |
| `codes.npy` | `(N, M)` uint8/16/32 — the compressed index |
| `docs.jsonl` | `{"id", "text", "metadata", "alive"}` per line |
| `embeddings.f16.npy` | *(optional)* float16 originals for reranking |

---

## 📖 API reference

### `train_corpus(corpus_npy, input_dim, ...) -> str`

| Param | Default | Description |
|---|---|---|
| `epochs` | 10 | training epochs |
| `num_subquantizers` | 8 | M — bytes/doc at K ≤ 256 |
| `codes_per_subquantizer` | 256 | K — codebook size per subspace/stage |
| `quantizer` | `"pq"` | `"pq"` or `"rq"` |
| `use_rotation` | False | OPQ-style learned rotation |
| `rank_weight` | 0.0 | order-preserving triplet loss weight (try 0.2) |
| `commitment_beta` | 0.25 | VQ commitment loss weight |
| `store_embeddings` | True | save float16 originals for reranking |
| `device` | `"cuda"` | falls back to CPU when unavailable |

### `VecmanIndex`

| Method | Description |
|---|---|
| `VecmanIndex.load(dir, device="cpu")` | load a trained index |
| `add_texts(texts, metadatas)` / `add_vectors(vectors, texts, metadatas)` | incremental insert, returns ids |
| `search(query, k, filter, method, ann, rerank, hybrid, ...)` | main search (see examples) |
| `search_batch(queries, k, **kw)` | many queries, one embedding batch |
| `find_similar(doc_id, k)` | more-like-this from stored codes |
| `range_search(query, min_score)` | threshold selection |
| `update(id, text/vector/metadata)` / `delete(ids)` / `compact()` | maintenance |
| `save(dir)` / `to(device)` | persistence / GPU |

`search()` returns `SearchResult(id, text, score, metadata)` ordered by descending score.

Rules of thumb: more subquantizers → better recall, more bytes/doc. Tiny corpora (< 1k docs) train better with a smaller config (`num_subquantizers=4, codes_per_subquantizer=16`) and many epochs (cheap — one step per epoch).

---

## 🔬 Evaluation with RAGAS

[`evaluate_webquestions.py`](evaluate_webquestions.py) builds a corpus from the Web Questions **train** split and evaluates on the held-out **test** split (no leakage), scoring faithfulness, answer relevancy, context precision/recall and answer correctness. Requires `vecman[eval]` and a `GOOGLE_API_KEY`.

## 🔧 Troubleshooting

| Symptom | Fix |
|---|---|
| All scores ≈ equal / same results for every query | Undertrained on a tiny corpus — raise `epochs` (hundreds are cheap when N is small), shrink `num_subquantizers`/`codes_per_subquantizer` |
| Low recall on near-duplicate documents | Expected at 8 bytes/doc — enable `rerank=True` or raise `num_subquantizers` |
| `OMP Error #15` / hard crash on Anaconda | Duplicate OpenMP runtimes; VECMAN routes its own matmuls through torch — do the same in your surrounding code, or set `KMP_DUPLICATE_LIB_OK=TRUE` |
| Rerank raises `RuntimeError` | Index was built with `store_embeddings=False` |
| Pre-3.0 index fails to load | The old format was lossy and unsafe — retrain (see [CHANGELOG](CHANGELOG.md)) |

## 🛠️ Development

```bash
git clone https://github.com/Vec1man/vecman.git && cd vecman
pip install -e ".[dev]" ruff
ruff check vecman/ tests/ benchmarks/
pytest tests/            # 51 tests, no network needed
```

CI runs lint + tests on Python 3.10/3.12 plus a benchmark smoke-run. See [CHANGELOG.md](CHANGELOG.md) for release history.

## 📖 Citation

```bibtex
@software{vecman,
  title={VECMAN: Learned Embedding Compression and Compressed Vector Search},
  author={Loaii abdalslam},
  year={2026},
  url={https://github.com/Vec1man/vecman}
}
```

## 📄 License

MIT — see [LICENSE](LICENSE).

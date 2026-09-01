# VECMAN

![VECMAN Logo](media/VV.png)

**Learned embedding compression and compressed vector search.**

VECMAN trains a product-quantized VQ-VAE on your text embeddings and stores
each document as a handful of bytes, then searches **directly in the
compressed latent space** — the corpus is never decompressed per query and
never re-embedded.

With the defaults (8 subquantizers × 256 codes), a 384-dim float32 embedding
(1,536 bytes) becomes **8 bytes**: a 192× storage reduction.

VECMAN is not a hosted database like Pinecone or Qdrant — it is a
compression + retrieval layer you can run anywhere PyTorch runs, with an
optional RAG layer (Gemini) and RAGAS evaluation tooling on top.

## How it works

```
text ──sentence-transformers──▶ embedding (384-d float32)
embedding ──encoder──▶ latent (96-d) ──product quantizer──▶ 8 codes (8 bytes)

query ──encoder──▶ latent ──cosine vs decompressed corpus latents──▶ top-k
```

- **Product quantization (PQ)**: the latent is split into M subspaces, each
  with its own learned codebook — the same idea behind FAISS-PQ, but with
  codebooks trained end-to-end by a neural encoder/decoder. **Residual
  quantization** (`quantizer="rq"`) and an **OPQ-style learned rotation**
  (`use_rotation=True`) are available for higher accuracy at the same bytes.
- **EMA codebook updates + dead-code reinitialization** prevent codebook
  collapse; optional **ranking loss** (`rank_weight`) trains top-k ordering
  directly.
- Corpus codes are decompressed **once** at load time; every query after
  that is one encoder pass plus one matrix multiplication — or, with
  **ADC**, a small lookup table and M gathers per document, so the latent
  matrix is never materialized.
- Candidate generation scales via **IVF** (spherical k-means++, auto above
  10k docs) or a dependency-free **HNSW** graph.
- **Two-stage reranking** (`rerank=True`) re-scores top candidates against
  float16 originals: near-exact recall at compressed-index speed.

## Installation

```bash
pip install vecman            # core: compression + search
pip install vecman[rag]       # + Gemini answer generation
pip install vecman[eval]      # + RAGAS evaluation
```

## Quick start

```python
import numpy as np
from vecman import VecmanIndex, embed_texts, save_jsonl, train_corpus

texts = ["Machine learning is...", "Deep learning uses...", ...]

# 1. Embed and train (one-time)
embeddings = embed_texts(texts)
np.save("index/corpus.npy", embeddings)
save_jsonl(texts, "index/docs.jsonl")
train_corpus("index/corpus.npy", input_dim=embeddings.shape[1],
             epochs=10, device="cpu", output_dir="index")

# 2. Load and search
index = VecmanIndex.load("index")
results = index.search("What is machine learning?", k=5)
for r in results:
    print(f"[{r.score:.3f}] {r.text}")

# 3. CRUD — no retraining needed
index.add_texts(["A new document"], metadatas=[{"lang": "en"}])
index.search("new stuff", k=3, filter={"lang": "en"})
index.delete(0)
index.save("index")

# 4. Advanced search
index.search("query", k=5, rerank=True)             # near-exact via stored f16 originals
index.search("query", k=5, hybrid=True)             # BM25 + dense, RRF fusion
index.search("query", k=5, method="adc")            # lookup-table scoring, lowest RAM
index.search("query", k=5, ann="hnsw")              # graph ANN
index.search("query", filter={"year": {"$gte": 2024}, "lang": {"$in": ["en", "ar"]}})
index.search_batch(["q1", "q2", "q3"], k=5)         # batched query embedding
index.find_similar(doc_id=7, k=5)                    # more-like-this from stored codes
index.range_search("query", min_score=0.6)          # similarity threshold selection
index.to("cuda")                                     # GPU search math
```

Or from the command line:

```bash
vecman index docs.txt -o my_index --epochs 10
vecman query "what is machine learning" -d my_index -k 5
vecman info -d my_index
vecman serve -d my_index -p 8080   # REST API, stdlib only
# GET  /search?q=...&k=5&rerank=1&hybrid=1     GET /info    GET /health
# POST /search {"queries": [...]}  POST /add  POST /delete  POST /save
```

## LangChain

```python
from vecman.integrations.langchain import VecmanVectorStore  # needs langchain-core

store = VecmanVectorStore(index, rerank=True)
store.similarity_search("query", k=5)
```

## RAG in one call

```python
from vecman import generate_answer  # requires vecman[rag] + GOOGLE_API_KEY

contexts = [r.text for r in index.search(question, k=5)]
answer = generate_answer(question, contexts)
```

## Configuration

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `num_subquantizers` (M) | 8 | PQ subspaces = bytes per document (at K ≤ 256) |
| `codes_per_subquantizer` (K) | 256 | Codebook size per subspace |
| `epochs` | 10 | Training epochs |
| `commitment_beta` | 0.25 | VQ commitment loss weight |
| `hidden_dim` | 1024 | Encoder/decoder MLP width |
| `nprobe` (search) | 8 | IVF clusters scanned per query |

Rules of thumb: more subquantizers → better recall, more bytes/doc. Tiny
corpora (< 1k docs) work better with a smaller config, e.g. `M=4, K=16`.

## Benchmarks

Run the reproducible benchmark yourself — it measures recall@k against
exact float32 cosine search, compression ratio, and query latency:

```bash
python benchmarks/benchmark.py --n 5000 --dim 384 --epochs 8 --k 10
```

We only publish numbers that come out of this script. One measured run
(CPU, synthetic clustered embeddings, v3.0.0):

```bash
python benchmarks/benchmark.py --n 2000 --dim 128 --epochs 40 --queries 50 --k 10 --clusters 200 --batch-size 256
```

| Metric | PQ | RQ (+rank loss) |
|--------|-----|-----|
| recall@10, compressed only | 0.83 | 0.84 |
| recall@10, ADC scoring | 0.83 (identical) | 0.84 (identical) |
| **recall@10 with rerank** | **0.98** | **0.99** |
| bytes/doc (codes) | 8 (vs 512 raw float32) | 8 |
| compression ratio | 64x | 64x |
| avg query, compressed | 0.7 ms | 2.8 ms |
| avg query, reranked | 0.7 ms | 3.0 ms |

Recall depends heavily on data geometry: distinguishing near-duplicate
documents from 8-byte codes is fundamentally hard (increase
`num_subquantizers` for finer resolution), while topical retrieval —
the typical RAG case — holds up well. Community benchmark results (with
the exact command used) are welcome as issues or PRs.

## Evaluation with RAGAS

`evaluate_webquestions.py` builds a corpus from the Web Questions **train**
split and evaluates on the held-out **test** split (no leakage), scoring
faithfulness, answer relevancy, context precision/recall and answer
correctness. Requires `vecman[eval]` and a `GOOGLE_API_KEY`.

## When (not) to use VECMAN

Use it when embedding storage is your bottleneck and you can accept a small
recall trade-off for a ~100–200× smaller index. If you need exact search on
a small corpus, use plain cosine similarity; if you need a managed,
distributed database, use Qdrant/Milvus/pgvector — and consider VECMAN-style
learned PQ as the compression layer inside it.

## Development

```bash
git clone https://github.com/Vec1man/vecman.git
cd vecman
pip install -e ".[dev]"
pytest tests/
```

See [CHANGELOG.md](CHANGELOG.md) for what changed in v3 (breaking: pre-3.0
indexes must be re-trained — the old storage format was lossy and unsafe).

## Citation

```bibtex
@software{vecman,
  title={VECMAN: Learned Embedding Compression and Compressed Vector Search},
  author={Loaii abdalslam},
  year={2026},
  url={https://github.com/Vec1man/vecman}
}
```

## License

MIT — see [LICENSE](LICENSE).

# Changelog

## 3.1.0 — 2026-09-01

Backward compatible with 3.0.0 indexes and checkpoints (new features are
opt-in; defaults unchanged).

### Added — representation & training
- **Residual quantization** (`quantizer="rq"`): staged full-dimension
  codebooks, usually more accurate than PQ at the same byte budget and
  hierarchical (a code prefix is already a coarse approximation).
- **OPQ-style learned rotation** (`use_rotation=True`): an orthogonal
  transform before quantization spreads variance across subspaces.
- **Order-preserving ranking loss** (`rank_weight`): sampled triplets
  penalize latent orderings that contradict confident input orderings —
  targets top-k quality directly.

### Added — search
- **ADC (asymmetric distance computation)** scoring (`method="adc"`): the
  query builds an (M, K) lookup table against the codebooks; document
  scores are M lookups each. Score-identical to the latent path, without
  ever materializing the latent matrix. Auto-selected above 50k docs.
- **Two-stage reranking** (`rerank=True`): original embeddings are stored
  as float16 (embeddings.f16.npy) and the top candidates are re-scored
  exactly. Measured on the bundled benchmark: recall@10 0.83 → 0.98 at
  64x compression, +0.04 ms/query.
- **HNSW** graph ANN (`ann="hnsw"`), dependency-free implementation with
  heuristic neighbour selection; IVF now seeds with k-means++.
- **Hybrid search** (`hybrid=True`): BM25 keyword scores fused with dense
  scores via reciprocal-rank fusion (`keyword_query` for vector queries).
- **Vector-space selection ops**: `search_batch` (batched query
  embedding), `find_similar` (more-like-this from stored codes),
  `range_search` (similarity threshold selection).
- **Rich metadata filters**: `$in`, `$nin`, `$ne`, `$gt`, `$gte`, `$lt`,
  `$lte`, `$contains` alongside plain equality.
- **GPU search** (`device="cuda"` on VecmanIndex / `.to()`); all search
  math runs through torch on the chosen device.

### Added — tooling
- `index.compact()` physically removes soft-deleted rows.
- REST API write endpoints: `POST /search` (batched), `/add`, `/delete`,
  `/save`; GET /search gains `rerank`/`hybrid` flags. CLI gains
  `--quantizer`, `--rotation`, `--rerank`, `--hybrid`.
- LangChain `VectorStore` adapter (`vecman.integrations.langchain`),
  lazy — no hard dependency.
- Ruff linting (config + CI step); 22 new tests (51 total).

## 3.0.0 — 2026-09-01

Complete rework. **Breaking**: models and indexes trained with pre-3.0
versions must be re-trained (the old storage format was lossy and unsafe —
see below).

### Fixed
- **Retrieval now actually uses the compressed codes.** Previously the
  stored codes were ignored and the entire corpus was re-embedded with
  sentence-transformers on every query. Codes are now decompressed once at
  load time; each query is one encoder pass + one matrix multiplication.
- **Integer overflow in code storage.** Codes were always written as
  `uint16`; codebooks larger than 65,536 entries silently wrapped around.
  Storage dtype is now chosen from the codebook size (`uint8`/`uint16`/`uint32`).
- **Evaluation data leakage.** The RAGAS evaluation script built its corpus
  and its evaluation questions from the same `train` split (and the corpus
  documents contained the answers). Evaluation questions now come from the
  held-out `test` split.
- **Silent error swallowing.** Broad `try/except` blocks that returned the
  first k documents with 0.0 scores as a "fallback" were removed; invalid
  input and internal failures now raise.
- Embedding model is loaded once per process and cached (was reloaded on
  every call). Documents are encoded in batches, not one at a time.
- The best (lowest-loss) model state is now actually kept and saved.

### Added
- **Product quantization**: the latent is split into M subspaces (default 8)
  each with its own 256-entry codebook — 8 bytes/document instead of one
  lossy code, 192x smaller than raw float32 384-dim embeddings.
- **EMA codebook updates + dead-code reinitialization** to prevent codebook
  collapse.
- **`VecmanIndex`**: incremental `add_texts`/`add_vectors` (no retrain),
  `delete`, `update`, metadata equality filters on `search`, `save`/`load`.
- **IVF index** (spherical k-means) automatically used above 10k documents;
  `nprobe` controls the accuracy/speed trade-off.
- **CLI**: `vecman index <docs>`, `vecman query "..."`, `vecman info`.
- **Test suite** (pytest) and GitHub Actions CI.
- **Reproducible benchmark** (`benchmarks/benchmark.py`) reporting recall@k
  vs exact search, compression ratio, and query latency.
- Packaging moved to `pyproject.toml` with optional extras: `vecman[rag]`
  (Gemini answer generation), `vecman[eval]` (RAGAS), `vecman[dev]`.

### Changed
- `generate_answer` moved to `vecman.rag` (still re-exported from `vecman`);
  the compression core no longer imports LLM or embedding dependencies.
- `latent_bits` is deprecated; use `num_subquantizers` /
  `codes_per_subquantizer`. Passing it maps to an equivalent configuration
  with a `DeprecationWarning`.
- Version is now defined in one place (`vecman.__version__` / pyproject),
  previously 2.7.10 in setup.py vs 0.1.0 in the package.
- Removed the committed `.env` file; README rewritten with measured claims
  only.

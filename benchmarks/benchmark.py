#!/usr/bin/env python3
"""Reproducible VECMAN benchmark: compressed search vs exact float32 search.

Measures, on synthetic clustered embeddings (no downloads needed):
  * recall@k of the compressed index against exact cosine ground truth
  * storage: bytes/document compressed vs raw float32
  * query latency for both paths

Usage:
    python benchmarks/benchmark.py --n 5000 --dim 384 --epochs 8 --k 10
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from vecman.core.index import VecmanIndex
from vecman.models.vqvae import VQVAE
from vecman.utils.training import train_corpus


def make_corpus(n: int, dim: int, n_clusters: int, seed: int = 0) -> np.ndarray:
    """Clustered Gaussian vectors, roughly mimicking topical text embeddings."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_clusters, dim)).astype(np.float32) * 2.0
    assignment = rng.integers(0, n_clusters, size=n)
    x = centers[assignment] + rng.normal(size=(n, dim)).astype(np.float32) * 0.5
    return x.astype(np.float32)


def exact_topk(corpus: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    # torch matmul, matching the library: keeps all BLAS work in one runtime.
    dots = (torch.from_numpy(corpus) @ torch.from_numpy(query)).numpy()
    norms = np.linalg.norm(corpus, axis=1) * np.linalg.norm(query)
    norms[norms < 1e-12] = 1e-12
    return np.argsort(dots / norms)[::-1][:k]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--clusters", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--subquantizers", type=int, default=8)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--quantizer", default="pq", choices=["pq", "rq"])
    parser.add_argument("--rank-weight", type=float, default=0.0)
    parser.add_argument("--rotation", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="benchmark_results.json")
    args = parser.parse_args()

    print(f"Corpus: {args.n} docs x {args.dim} dims, {args.clusters} clusters")
    corpus = make_corpus(args.n, args.dim, args.clusters)
    work = Path("benchmark_work")
    work.mkdir(exist_ok=True)
    np.save(work / "corpus.npy", corpus)

    t0 = time.perf_counter()
    train_corpus(
        str(work / "corpus.npy"), input_dim=args.dim, epochs=args.epochs,
        num_subquantizers=args.subquantizers, device=args.device,
        output_dir=str(work), batch_size=args.batch_size,
        quantizer=args.quantizer, rank_weight=args.rank_weight,
        use_rotation=args.rotation,
    )
    train_time = time.perf_counter() - t0

    meta = json.loads((work / "vqvae_meta.json").read_text())
    model = VQVAE.from_config(meta)
    model.load_state_dict(torch.load(work / "vqvae.pt", map_location="cpu"))
    model.eval()

    index = VecmanIndex(model)
    index.add_vectors(corpus, [f"doc-{i}" for i in range(args.n)])
    index._ensure_latents()  # exclude one-time decompression from latency

    # Queries: perturbed corpus vectors (so ground truth is meaningful).
    rng = np.random.default_rng(1)
    q_ids = rng.choice(args.n, size=args.queries, replace=False)
    queries = corpus[q_ids] + rng.normal(
        size=(args.queries, args.dim)
    ).astype(np.float32) * 0.1

    recalls, rerank_recalls, adc_recalls = [], [], []
    compressed_times, rerank_times, exact_times = [], [], []
    for q in queries:
        t0 = time.perf_counter()
        truth = set(exact_topk(corpus, q, args.k).tolist())
        exact_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        results = index.search(q, k=args.k, method="latent")
        compressed_times.append(time.perf_counter() - t0)
        recalls.append(len(truth & {r.id for r in results}) / args.k)

        adc = index.search(q, k=args.k, method="adc")
        adc_recalls.append(len(truth & {r.id for r in adc}) / args.k)

        t0 = time.perf_counter()
        reranked = index.search(q, k=args.k, rerank=True)
        rerank_times.append(time.perf_counter() - t0)
        rerank_recalls.append(len(truth & {r.id for r in reranked}) / args.k)

    bytes_per_doc = index.codes.dtype.itemsize * index.codes.shape[1]
    raw_bytes = args.dim * 4
    results = {
        "n_docs": args.n,
        "dim": args.dim,
        "k": args.k,
        "quantizer": args.quantizer,
        "recall_at_k_compressed": float(np.mean(recalls)),
        "recall_at_k_adc": float(np.mean(adc_recalls)),
        "recall_at_k_reranked": float(np.mean(rerank_recalls)),
        "bytes_per_doc_compressed": bytes_per_doc,
        "bytes_per_doc_raw_float32": raw_bytes,
        "compression_ratio": raw_bytes / bytes_per_doc,
        "avg_query_ms_compressed": float(np.mean(compressed_times) * 1000),
        "avg_query_ms_reranked": float(np.mean(rerank_times) * 1000),
        "avg_query_ms_exact": float(np.mean(exact_times) * 1000),
        "train_seconds": train_time,
    }

    print("\n=== Benchmark results ===")
    for key, value in results.items():
        print(f"{key:28s}: {value:.4f}" if isinstance(value, float) else f"{key:28s}: {value}")
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()

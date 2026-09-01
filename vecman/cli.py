"""VECMAN command-line interface.

Usage::

    vecman index docs.txt -o my_index --epochs 10 --device cpu
    vecman query "what is machine learning" -d my_index -k 5
    vecman info -d my_index
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _read_documents(path: Path) -> list:
    """Read documents from a .txt (one per line), .jsonl ({"text": ...}),
    or a directory of .txt/.md files (one document per file)."""
    if path.is_dir():
        docs = []
        for file in sorted(path.rglob("*")):
            if file.suffix.lower() in (".txt", ".md") and file.is_file():
                content = file.read_text(encoding="utf-8", errors="replace").strip()
                if content:
                    docs.append(content)
        return docs
    if path.suffix == ".jsonl":
        docs = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(json.loads(line)["text"])
        return docs
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    return [line.strip() for line in text.splitlines() if line.strip()]


def cmd_index(args: argparse.Namespace) -> int:
    from .core.index import VecmanIndex
    from .utils.embedding import embed_texts
    from .utils.retrieval import save_jsonl
    from .utils.training import train_corpus

    source = Path(args.source)
    if not source.exists():
        print(f"error: {source} does not exist", file=sys.stderr)
        return 1

    docs = _read_documents(source)
    if not docs:
        print(f"error: no documents found in {source}", file=sys.stderr)
        return 1
    print(f"Read {len(docs)} documents from {source}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Embedding with {args.embedding_model}...")
    embeddings = embed_texts(docs, args.embedding_model)
    corpus_path = out_dir / "corpus.npy"
    np.save(corpus_path, embeddings)

    save_jsonl(docs, str(out_dir / "docs.jsonl"))
    train_corpus(
        str(corpus_path),
        input_dim=embeddings.shape[1],
        epochs=args.epochs,
        num_subquantizers=args.subquantizers,
        device=args.device,
        output_dir=str(out_dir),
        embedding_model=args.embedding_model,
    )

    index = VecmanIndex.load(out_dir)
    print(f"Index ready in {out_dir} ({len(index)} documents)")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    from .core.index import VecmanIndex

    index = VecmanIndex.load(args.dir)
    filter_dict = json.loads(args.filter) if args.filter else None
    results = index.search(args.query, k=args.k, filter=filter_dict)
    if not results:
        print("(no results)")
        return 0
    for rank, r in enumerate(results, 1):
        snippet = r.text.replace("\n", " ")
        if len(snippet) > 160:
            snippet = snippet[:157] + "..."
        print(f"{rank}. [{r.score:.3f}] (id={r.id}) {snippet}")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    meta_path = Path(args.dir) / "vqvae_meta.json"
    if not meta_path.exists():
        print(f"error: {meta_path} not found", file=sys.stderr)
        return 1
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    for key, value in meta.items():
        print(f"{key:26s}: {value}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vecman",
        description="VECMAN: learned embedding compression and compressed vector search",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="embed, train and build an index from documents")
    p_index.add_argument("source", help=".txt (one doc/line), .jsonl, or a directory of .txt/.md files")
    p_index.add_argument("-o", "--output", default="vecman_index", help="output directory")
    p_index.add_argument("--epochs", type=int, default=10)
    p_index.add_argument("--subquantizers", type=int, default=8, help="PQ subspaces (bytes per document)")
    p_index.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p_index.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    p_index.set_defaults(func=cmd_index)

    p_query = sub.add_parser("query", help="search an index")
    p_query.add_argument("query")
    p_query.add_argument("-d", "--dir", default="vecman_index", help="index directory")
    p_query.add_argument("-k", type=int, default=5)
    p_query.add_argument("--filter", help='metadata filter as JSON, e.g. \'{"lang": "en"}\'')
    p_query.set_defaults(func=cmd_query)

    p_info = sub.add_parser("info", help="show index metadata")
    p_info.add_argument("-d", "--dir", default="vecman_index")
    p_info.set_defaults(func=cmd_info)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

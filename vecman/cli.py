"""VECMAN command-line interface.

Usage::

    vecman index docs.txt -o my_index --epochs 10 --device cpu
    vecman query "what is machine learning" -d my_index -k 5
    vecman info -d my_index
    vecman serve -d my_index -p 8080   # REST API (stdlib, no extra deps)
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
        quantizer=args.quantizer,
        use_rotation=args.rotation,
    )

    index = VecmanIndex.load(out_dir)
    print(f"Index ready in {out_dir} ({len(index)} documents)")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    from .core.index import VecmanIndex

    index = VecmanIndex.load(args.dir)
    filter_dict = json.loads(args.filter) if args.filter else None
    results = index.search(args.query, k=args.k, filter=filter_dict,
                           rerank=args.rerank, hybrid=args.hybrid)
    if not results:
        print("(no results)")
        return 0
    for rank, r in enumerate(results, 1):
        snippet = r.text.replace("\n", " ")
        if len(snippet) > 160:
            snippet = snippet[:157] + "..."
        print(f"{rank}. [{r.score:.3f}] (id={r.id}) {snippet}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Serve an index over HTTP using only the standard library.

    Endpoints:
        GET  /search?q=<text>&k=<int>&rerank=1&hybrid=1
                                       -> {"results": [{id, text, score, metadata}]}
        POST /search  {"queries": [...], "k": 5, "filter": {...}, "rerank": false}
                                       -> {"results": [[...], ...]}  (batched)
        POST /add     {"texts": [...], "metadatas": [...]} -> {"ids": [...]}
        POST /delete  {"ids": [...]}   -> {"removed": n}
        POST /save    {}               -> {"saved": true}
        GET  /info                     -> index metadata
        GET  /health                   -> {"status": "ok"}
    """
    import urllib.parse
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from .core.index import VecmanIndex

    index = VecmanIndex.load(args.dir)
    index._ensure_latents()  # decompress once, before serving traffic
    print(f"Loaded index with {len(index)} documents from {args.dir}")

    def _serialize(results) -> list:
        return [
            {"id": r.id, "text": r.text, "score": r.score,
             "metadata": r.metadata}
            for r in results
        ]

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/health":
                self._send(200, {"status": "ok"})
                return
            if parsed.path == "/info":
                self._send(200, {"documents": len(index), **index.model.config()})
                return
            if parsed.path == "/search":
                params = urllib.parse.parse_qs(parsed.query)
                query = (params.get("q") or [""])[0]
                if not query.strip():
                    self._send(400, {"error": "missing query parameter 'q'"})
                    return
                try:
                    k = int((params.get("k") or ["5"])[0])
                    rerank = (params.get("rerank") or ["0"])[0] == "1"
                    hybrid = (params.get("hybrid") or ["0"])[0] == "1"
                    results = index.search(query, k=k, rerank=rerank,
                                           hybrid=hybrid)
                except (ValueError, RuntimeError, KeyError) as e:
                    self._send(400, {"error": str(e)})
                    return
                self._send(200, {"results": _serialize(results)})
                return
            self._send(404, {"error": f"unknown path {parsed.path}"})

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def do_POST(self) -> None:  # noqa: N802 (stdlib naming)
            parsed = urllib.parse.urlparse(self.path)
            try:
                body = self._read_json()
            except (ValueError, json.JSONDecodeError) as e:
                self._send(400, {"error": f"invalid JSON body: {e}"})
                return
            try:
                if parsed.path == "/search":
                    queries = body.get("queries") or []
                    if not queries:
                        self._send(400, {"error": "missing 'queries' list"})
                        return
                    batches = index.search_batch(
                        queries, k=int(body.get("k", 5)),
                        filter=body.get("filter"),
                        rerank=bool(body.get("rerank", False)),
                    )
                    self._send(200, {"results": [_serialize(b) for b in batches]})
                elif parsed.path == "/add":
                    texts = body.get("texts") or []
                    if not texts:
                        self._send(400, {"error": "missing 'texts' list"})
                        return
                    ids = index.add_texts(texts, metadatas=body.get("metadatas"))
                    self._send(200, {"ids": ids})
                elif parsed.path == "/delete":
                    removed = index.delete([int(i) for i in body.get("ids", [])])
                    self._send(200, {"removed": removed})
                elif parsed.path == "/save":
                    index.save(args.dir)
                    self._send(200, {"saved": True})
                else:
                    self._send(404, {"error": f"unknown path {parsed.path}"})
            except (ValueError, RuntimeError, KeyError) as e:
                self._send(400, {"error": str(e)})

        def log_message(self, fmt: str, *log_args) -> None:
            print(f"{self.address_string()} - {fmt % log_args}")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving on http://{args.host}:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
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
    p_index.add_argument("--quantizer", default="pq", choices=["pq", "rq"],
                         help="pq: subspace codebooks; rq: residual stages")
    p_index.add_argument("--rotation", action="store_true",
                         help="learn an OPQ-style rotation before quantization")
    p_index.set_defaults(func=cmd_index)

    p_query = sub.add_parser("query", help="search an index")
    p_query.add_argument("query")
    p_query.add_argument("-d", "--dir", default="vecman_index", help="index directory")
    p_query.add_argument("-k", type=int, default=5)
    p_query.add_argument("--filter", help='metadata filter as JSON, e.g. \'{"lang": "en"}\'')
    p_query.add_argument("--rerank", action="store_true",
                         help="rerank against stored original embeddings")
    p_query.add_argument("--hybrid", action="store_true",
                         help="fuse BM25 keyword scores with dense scores")
    p_query.set_defaults(func=cmd_query)

    p_info = sub.add_parser("info", help="show index metadata")
    p_info.add_argument("-d", "--dir", default="vecman_index")
    p_info.set_defaults(func=cmd_info)

    p_serve = sub.add_parser("serve", help="serve an index over a REST API")
    p_serve.add_argument("-d", "--dir", default="vecman_index", help="index directory")
    p_serve.add_argument("-p", "--port", type=int, default=8080)
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.set_defaults(func=cmd_serve)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

"""VECMAN v3 example: train, index, search, and (optionally) generate.

Run with:  python example_usage.py
"""

import os

import numpy as np

from vecman import VecmanIndex, embed_texts, generate_answer, save_jsonl, train_corpus

TEXTS = [
    "Machine learning is a subset of artificial intelligence that enables computers to learn from experience.",
    "Deep learning uses neural networks with multiple layers to model complex patterns in data.",
    "Natural language processing helps computers understand, interpret, and generate human language.",
    "Computer vision enables machines to interpret and analyze visual information.",
    "Reinforcement learning trains agents through rewards and penalties to make sequential decisions.",
    "Supervised learning uses labeled training data to learn a mapping from inputs to outputs.",
    "Unsupervised learning finds hidden patterns in data without labeled examples.",
    "Transfer learning leverages pre-trained models to solve new but related problems with less data.",
    "Feature engineering selects and transforms variables for machine learning models.",
    "Cross-validation assesses how well a model generalizes to independent datasets.",
    "Clustering and dimensionality reduction are key unsupervised learning techniques.",
    "Classification and regression are common supervised learning problem types.",
]


def main() -> None:
    work_dir = "example_index"

    # 1. Embed the corpus and save training artifacts.
    print("1. Embedding corpus...")
    embeddings = embed_texts(TEXTS)
    os.makedirs(work_dir, exist_ok=True)
    corpus_path = os.path.join(work_dir, "corpus.npy")
    np.save(corpus_path, embeddings)
    save_jsonl(TEXTS, os.path.join(work_dir, "docs.jsonl"))

    # 2. Train the product-quantized VQ-VAE. With M=4 subquantizers each
    #    document is stored as 4 bytes (vs 1536 bytes of raw float32).
    print("2. Training VQ-VAE...")
    train_corpus(
        corpus_path,
        input_dim=embeddings.shape[1],
        epochs=400,                # tiny corpus -> one step per epoch, so
        num_subquantizers=4,       # many epochs are cheap and necessary
        codes_per_subquantizer=16,
        device="cpu",
        output_dir=work_dir,
        batch_size=64,
        hidden_dim=256,
    )

    # 3. Load the index. Codes are decompressed ONCE here; queries after
    #    this are a single encoder pass + one matrix multiplication.
    print("3. Loading index...")
    index = VecmanIndex.load(work_dir)
    print(f"   {len(index)} documents, {index.codes.shape[1]} bytes each")

    # 4. Search — including an incremental add and a metadata filter.
    index.add_texts(
        ["Gradient descent iteratively minimizes a loss function."],
        metadatas=[{"topic": "optimization"}],
    )
    print("4. Searching...")
    for question in [
        "What is machine learning?",
        "How do neural networks work?",
        "How are models optimized?",
    ]:
        print(f"\n   Q: {question}")
        for rank, result in enumerate(index.search(question, k=3), 1):
            print(f"   {rank}. [{result.score:.3f}] {result.text[:80]}")

    # 5. Optional: RAG answer generation (requires GOOGLE_API_KEY and
    #    `pip install vecman[rag]`).
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        question = "What is machine learning?"
        contexts = [r.text for r in index.search(question, k=3)]
        print(f"\n5. Gemini answer:\n   {generate_answer(question, contexts, api_key=api_key)}")
    else:
        print("\n5. Skipping answer generation (GOOGLE_API_KEY not set)")


if __name__ == "__main__":
    main()

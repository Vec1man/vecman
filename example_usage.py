#!/usr/bin/env python3
"""
VECMAN Example Usage

This script demonstrates how to use the VECMAN package for text embedding,
training a VQ-VAE model, and performing retrieval-augmented generation.
"""

import os
import numpy as np
from pathlib import Path
from vecman import VQVAE, train_corpus, embed_texts, save_jsonl, load_assets, retrieve, generate_answer

def main():
    # Example data - more diverse examples for better training + synthetic variations
    base_texts = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience.",
        "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.",
        "Natural language processing helps computers understand, interpret, and generate human language in a valuable way.",
        "Computer vision enables machines to interpret and analyze visual information from the world around them.",
        "Reinforcement learning trains agents through rewards and penalties to make sequential decisions in an environment.",
        "Supervised learning uses labeled training data to learn a mapping from inputs to outputs.",
        "Unsupervised learning finds hidden patterns in data without using labeled examples.",
        "Transfer learning leverages pre-trained models to solve new but related problems with less data.",
        "Feature engineering is the process of selecting and transforming variables for machine learning models.",
        "Cross-validation is a technique to assess how well a model will generalize to independent datasets."
    ]
    
    # Add synthetic variations to expand training data for better VQ-VAE learning
    synthetic_texts = [
        # ML variations
        "Artificial intelligence includes machine learning as a key component for learning from data.",
        "Machine learning algorithms improve performance through experience and data exposure.",
        "AI systems use machine learning to adapt and enhance their capabilities over time.",
        
        # Deep learning variations  
        "Neural networks with multiple hidden layers form the foundation of deep learning systems.",
        "Deep learning models excel at pattern recognition through layered neural architectures.",
        "Multi-layer neural networks enable deep learning to solve complex recognition tasks.",
        
        # NLP variations
        "Language models help computers process and understand human communication patterns.",
        "Text processing and natural language understanding are core NLP capabilities.",
        "Computational linguistics enables machines to work with human language effectively.",
        
        # Computer vision variations
        "Image recognition and visual analysis are primary goals of computer vision systems.",
        "Visual perception algorithms help machines understand and interpret image content.",
        "Computer vision systems process visual data to extract meaningful information.",
        
        # Supervised learning variations
        "Training with labeled examples enables supervised learning algorithms to make predictions.",
        "Supervised algorithms learn input-output mappings from annotated training datasets.",
        "Classification and regression are common supervised learning problem types.",
        
        # Unsupervised learning variations
        "Clustering and dimensionality reduction are key unsupervised learning techniques.",
        "Pattern discovery in unlabeled data is the main goal of unsupervised methods.",
        "Unsupervised algorithms identify hidden structures without labeled training examples."
    ]
    
    # Combine base and synthetic data for richer training
    texts = base_texts + synthetic_texts
    
    print("🚀 VECMAN Example Usage")
    print("=" * 50)
    
    try:
        # Step 1: Embed texts
        print("📝 Step 1: Embedding texts...")
        embeddings = embed_texts(texts)
        print(f"   Generated embeddings shape: {embeddings.shape}")
        
        # Step 2: Save embeddings and documents
        print("💾 Step 2: Saving data...")
        corpus_path = "example_corpus.npy"
        docs_path = "docs.jsonl"
        
        np.save(corpus_path, embeddings)
        save_jsonl(texts, docs_path)
        print(f"   Saved: {corpus_path}, {docs_path}")
        
        # Step 3: Train VQ-VAE
        print("🏋️ Step 3: Training VQ-VAE...")
        output_dir = train_corpus(
            corpus_path, 
            input_dim=embeddings.shape[1], 
            epochs=10,  # Increased epochs
            device="cpu",  # Using CPU for compatibility
            latent_bits=20,  # Increased from 12 to 20 for better representation
            batch_size=min(8192, len(texts) * 10),  # Increased batch size, but limit for small datasets
            learning_rate=1e-3,  # Increased learning rate
            commitment_beta=0.1  # Lower commitment loss for less quantization pressure
        )
        print(f"   Training completed! Output dir: {output_dir}")
        
        # Step 4: Load trained model
        print("📂 Step 4: Loading trained model...")
        vqvae, codes, docs = load_assets(output_dir)
        print(f"   Loaded model with {len(docs)} documents")
        
        # Step 5: Perform retrieval with similarity scores - PURE VQ-VAE ONLY
        print("🔍 Step 5: Testing PURE VQ-VAE retrieval (no semantic fallbacks)...")
        questions = [
            "What is machine learning?",
            "How do neural networks work?",
            "What is the difference between supervised and unsupervised learning?"
        ]
        
        for question in questions:
            print(f"\n   Query: {question}")
            
            # Get query embedding
            q_vec = embed_texts([question])[0]
            
            # ONLY VQ-VAE retrieval - no semantic fallbacks
            print("   🔧 VQ-VAE retrieval ONLY:")
            vqvae_docs, vqvae_scores = retrieve(vqvae, codes, docs, q_vec, k=3, method="vqvae", return_scores=True)
            for i, (doc, score) in enumerate(zip(vqvae_docs, vqvae_scores), 1):
                print(f"      {i}. [{score:.3f}] {doc[:100]}...")
        
        # Step 6: Optional - Generate answer using ONLY VQ-VAE
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            print("\n🤖 Step 6: Generating answer with PURE VQ-VAE context...")
            question = "What is machine learning?"
            q_vec = embed_texts([question])[0]
            
            # Use ONLY VQ-VAE method - no auto fallback
            context_docs, context_scores = retrieve(vqvae, codes, docs, q_vec, k=3, method="vqvae", return_scores=True)
            
            # Show which documents were selected with scores
            print(f"   📄 VQ-VAE selected context (avg score: {np.mean(context_scores):.3f}):")
            for i, (doc, score) in enumerate(zip(context_docs, context_scores), 1):
                print(f"      {i}. [{score:.3f}] {doc[:80]}...")
            
            # Use a more flexible prompt template
            custom_template = """
            You are a helpful assistant. Use the following information to answer the question.
            If you can find relevant information in the context, provide a clear and informative answer.
            If the context doesn't contain relevant information, say 'I don't have enough relevant information.'

            Context:
            {context}

            Question: {question}

            Answer:
            """
            
            try:
                answer = generate_answer(question, context_docs, api_key=api_key, prompt_template=custom_template)
                print(f"   Q: {question}")
                print(f"   A: {answer}")
            except Exception as e:
                print(f"   ⚠️ Could not generate answer: {e}")
        else:
            print("\n🤖 Step 6: Skipping answer generation (no GOOGLE_API_KEY)")
        
        print("\n✅ VECMAN example completed successfully!")
        
        # Cleanup example files
        cleanup_files = [corpus_path, docs_path]
        for file_path in cleanup_files:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"🧹 Cleaned up: {file_path}")
        
    except Exception as e:
        print(f"❌ Example failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Cleanup on error
        cleanup_files = ["example_corpus.npy", "docs.jsonl"]
        for file_path in cleanup_files:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"🧹 Cleaned up: {file_path}")
                except:
                    pass

if __name__ == "__main__":
    main() 
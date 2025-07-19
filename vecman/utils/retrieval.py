"""VECMAN retrieval utilities for VQ-VAE based vector database."""

import json
import os
from pathlib import Path
from typing import List, Tuple, Optional, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from ..models.vqvae import VQVAE

try:
    import google.generativeai as genai
except ImportError:
    genai = None

def embed_texts(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Embed texts using Sentence Transformers.
    
    Args:
        texts: List of texts to embed
        model_name: Name of the sentence-transformers model to use
        
    Returns:
        Array of embeddings with shape (len(texts), embedding_dim)
        
    Raises:
        ValueError: If texts list is empty or contains only empty strings
    """
    if not texts:
        raise ValueError("texts list cannot be empty")
    
    # Filter out empty texts and warn
    valid_texts = [t for t in texts if t and t.strip()]
    if not valid_texts:
        raise ValueError("All texts are empty or None")
    
    if len(valid_texts) != len(texts):
        print(f"⚠️ Filtered {len(texts) - len(valid_texts)} empty texts")
    
    try:
        model = SentenceTransformer(model_name)
        embeddings = model.encode(
            valid_texts,
            batch_size=512,
            convert_to_numpy=True,
            show_progress_bar=len(valid_texts) > 100
        ).astype("float32")
        
        # Handle case where original texts had empty entries
        if len(valid_texts) != len(texts):
            # Create full embedding array with zeros for empty texts
            full_embeddings = np.zeros((len(texts), embeddings.shape[1]), dtype=np.float32)
            valid_idx = 0
            for i, text in enumerate(texts):
                if text and text.strip():
                    full_embeddings[i] = embeddings[valid_idx]
                    valid_idx += 1
            return full_embeddings
        
        return embeddings
        
    except Exception as e:
        raise RuntimeError(f"Failed to embed texts with model {model_name}: {e}")

def save_jsonl(texts: List[str], path: str = "docs.jsonl"):
    """Save texts to JSONL format with IDs.
    
    Args:
        texts: List of texts to save
        path: Output path for JSONL file
        
    Raises:
        ValueError: If texts list is empty
        IOError: If file cannot be written
    """
    if not texts:
        raise ValueError("texts list cannot be empty")
    
    try:
        with open(path, "w", encoding="utf-8") as f:
            for i, t in enumerate(texts):
                # Handle None or empty texts
                text_content = str(t) if t is not None else ""
                json.dump({"id": i, "text": text_content}, f, ensure_ascii=False)
                f.write("\n")
        print(f"💾 Saved {len(texts)} documents to {path}")
    except Exception as e:
        raise IOError(f"Failed to save JSONL file {path}: {e}")

def load_assets(model_dir: Optional[str] = None) -> Tuple[VQVAE, np.ndarray, List[str]]:
    """Load trained VQ-VAE model and associated assets.
    
    Args:
        model_dir: Directory containing model artifacts (default: current directory)
        
    Returns:
        Tuple of (vqvae model, codes array, documents list)
        
    Raises:
        FileNotFoundError: If required files are missing
        ValueError: If files are corrupted or incompatible
    """
    model_dir = Path(model_dir) if model_dir else Path.cwd()
    
    # Check required files
    required_files = ["vqvae_meta.json", "vqvae.pt", "corpus.codes.bin", "docs.jsonl"]
    missing_files = [f for f in required_files if not (model_dir / f).exists()]
    
    if missing_files:
        raise FileNotFoundError(
            f"Missing required files in {model_dir}: {missing_files}"
        )
    
    try:
        # Load metadata
        with open(model_dir / "vqvae_meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        
        required_keys = ["input_dim", "latent_bits"]
        missing_keys = [k for k in required_keys if k not in meta]
        if missing_keys:
            raise ValueError(f"Missing metadata keys: {missing_keys}")
        
        # Initialize and load model
        vqvae = VQVAE(
            meta["input_dim"], 
            hidden=meta.get("hidden_dim", 1024),
            latent_bits=meta["latent_bits"]
        )
        
        # Load model state
        state_dict = torch.load(model_dir / "vqvae.pt", map_location="cpu")
        vqvae.load_state_dict(state_dict)
        vqvae.eval()
        
        # Load codes
        codes_path = model_dir / "corpus.codes.bin"
        codes = np.fromfile(codes_path, dtype=np.uint16)
        
        # Load documents
        docs = []
        with open(model_dir / "docs.jsonl", "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    doc_data = json.loads(line.strip())
                    if "text" not in doc_data:
                        print(f"⚠️ Line {line_num}: missing 'text' field")
                        docs.append("")
                    else:
                        docs.append(doc_data["text"])
                except json.JSONDecodeError as e:
                    print(f"⚠️ Line {line_num}: invalid JSON - {e}")
                    docs.append("")
        
        # Validate consistency
        if len(codes) != len(docs):
            print(f"⚠️ Size mismatch: {len(codes)} codes vs {len(docs)} docs")
            # Truncate to minimum length
            min_len = min(len(codes), len(docs))
            codes = codes[:min_len]
            docs = docs[:min_len]
        
        print(f"✅ Loaded VQ-VAE model with {len(docs)} documents")
        return vqvae, codes, docs
        
    except Exception as e:
        raise ValueError(f"Failed to load assets from {model_dir}: {e}")

def retrieve(vqvae: VQVAE,
            codes: np.ndarray,
            docs: List[str],
            q_vec: np.ndarray,
            k: int = 5,
            method: str = "auto",
            query_text: str = "",
            return_scores: bool = True) -> Union[List[str], Tuple[List[str], List[float]]]:
    """Enhanced retrieve with automatic fallback, multiple methods, and similarity scores.
    
    Args:
        vqvae: Trained VQ-VAE model
        codes: Array of document codes
        docs: List of document texts
        q_vec: Query vector
        k: Number of documents to retrieve
        method: 'auto', 'vqvae', 'semantic', or 'hybrid'
        query_text: Original query text (for semantic search)
        return_scores: Whether to return similarity scores along with documents
        
    Returns:
        List of retrieved document texts, or tuple of (documents, scores) if return_scores=True
        
    Raises:
        ValueError: If inputs are invalid
    """
    # Input validation
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    
    if len(docs) == 0:
        print("⚠️ No documents available for retrieval")
        return ([], []) if return_scores else []
    
    if len(codes) != len(docs):
        raise ValueError(f"Codes length {len(codes)} doesn't match docs length {len(docs)}")
    
    # Ensure k doesn't exceed available documents
    k = min(k, len(docs))
    
    # Validate query vector
    if q_vec is None or q_vec.size == 0:
        print("⚠️ Invalid query vector, using fallback")
        fallback_docs = docs[:k]
        fallback_scores = [0.0] * len(fallback_docs)
        return (fallback_docs, fallback_scores) if return_scores else fallback_docs
    
    if method == "semantic" and query_text:
        try:
            retrieved_docs, scores = semantic_retrieve_with_scores(docs, query_text, k)
            return (retrieved_docs, scores) if return_scores else retrieved_docs
        except Exception as e:
            print(f"⚠️ Semantic retrieval failed: {e}, using fallback")
            fallback_docs = docs[:k]
            fallback_scores = [0.0] * len(fallback_docs)
            return (fallback_docs, fallback_scores) if return_scores else fallback_docs
    
    elif method == "vqvae":
        try:
            retrieved_docs, scores = retrieve_vqvae(vqvae, codes, docs, q_vec, k)
            
            # NO FALLBACKS - return pure VQ-VAE results regardless of score quality
            print(f"🔧 Pure VQ-VAE results (max score: {np.max(scores) if len(scores) > 0 else 0:.4f})")
            return (retrieved_docs, scores.tolist()) if return_scores else retrieved_docs
            
        except Exception as e:
            print(f"⚠️ VQ-VAE retrieval failed: {e}")
            fallback_docs = docs[:k]
            fallback_scores = [0.0] * len(fallback_docs)
            return (fallback_docs, fallback_scores) if return_scores else fallback_docs
    
    else:  # method == "auto" or "hybrid"
        try:
            # Try VQ-VAE first
            retrieved_docs, scores = retrieve_vqvae(vqvae, codes, docs, q_vec, k)
            
            # Check quality with improved threshold - focus on best match
            if len(scores) > 0 and np.max(scores) > 0.05:  # At least one decent match
                return (retrieved_docs, scores.tolist()) if return_scores else retrieved_docs
            else:
                # VQ-VAE quality poor, try semantic if possible
                if query_text:
                    print(f"🔄 Using semantic fallback (VQ-VAE max score: {np.max(scores):.4f})")
                    retrieved_docs, scores = semantic_retrieve_with_scores(docs, query_text, k)
                    return (retrieved_docs, scores) if return_scores else retrieved_docs
                else:
                    print("⚠️ VQ-VAE scores low and no query text for semantic fallback")
                    fallback_docs = docs[:k]
                    fallback_scores = [0.0] * len(fallback_docs)
                    return (fallback_docs, fallback_scores) if return_scores else fallback_docs
        except Exception as e:
            print(f"⚠️ Auto retrieval failed: {e}, using fallback")
            fallback_docs = docs[:k]
            fallback_scores = [0.0] * len(fallback_docs)
            return (fallback_docs, fallback_scores) if return_scores else fallback_docs

def generate_answer(question: str,
                   context: List[str],
                   model: str = "gemini-2.0-flash",
                   api_key: Optional[str] = None,
                   prompt_template: Optional[str] = None) -> str:
    """Generate answer using Gemini Pro model.
    
    Args:
        question: User query
        context: List of retrieved documents for context
        model: Gemini model name
        api_key: Google API key (optional if already configured)
        prompt_template: Custom prompt template. Should include {context} and {question} placeholders.
                        If None, uses default template.
        
    Returns:
        Generated answer string
        
    Raises:
        RuntimeError: If google-generativeai package is not installed
        ValueError: If required arguments are missing
    """
    if genai is None:
        raise RuntimeError(
            "google-generativeai SDK is not installed. "
            "Run `pip install google-generativeai>=0.8`."
        )
    
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")
    
    if not context:
        return "I don't have enough context to answer this question."
    
    # Configure API key
    if api_key:
        genai.configure(api_key=api_key)
    elif not os.getenv("GOOGLE_API_KEY"):
        print("⚠️ No API key provided and GOOGLE_API_KEY not set")
        return "I need an API key to generate answers."
    
    # Use custom template or default
    if prompt_template is None:
        prompt_template = (
            "Answer the question using ONLY the context provided below. "
            "Be concise and factual. If the answer isn't in the context, "
            "say 'I don't know based on the provided context.'\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )
    
    try:
        # Format context for insertion
        formatted_context = "\n".join(f"[Doc {i+1}] {c}" for i, c in enumerate(context) if c.strip())
        
        if not formatted_context:
            return "I don't have enough relevant context to answer this question."
        
        # Generate final prompt
        prompt = prompt_template.format(context=formatted_context, question=question)
        
        # Generate response
        gm = genai.GenerativeModel(model)
        resp = gm.generate_content(prompt)
        
        if resp and resp.text:
            return resp.text.strip()
        else:
            return "I couldn't generate a response at this time."
            
    except Exception as e:
        print(f"⚠️ Error generating answer: {e}")
        return "I encountered an error while generating the answer."

def semantic_retrieve(docs: List[str], 
                     query: str, 
                     k: int = 5,
                     model_name: str = "all-MiniLM-L6-v2") -> List[str]:
    """Direct semantic retrieval using sentence transformers.
    
    Args:
        docs: List of documents to search
        query: Search query
        k: Number of documents to retrieve
        model_name: Name of the sentence transformer model
        
    Returns:
        List of top-k most similar documents
        
    Raises:
        ValueError: If inputs are invalid
    """
    if not docs:
        raise ValueError("Documents list cannot be empty")
    
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    
    k = min(k, len(docs))
    
    try:
        model = SentenceTransformer(model_name)
        
        # Encode query and documents
        query_emb = model.encode([query], convert_to_numpy=True)[0]
        
        # Filter out empty documents
        valid_docs = [(i, doc) for i, doc in enumerate(docs) if doc and doc.strip()]
        
        if not valid_docs:
            print("⚠️ No valid documents for semantic search")
            return docs[:k]
        
        valid_indices, valid_texts = zip(*valid_docs)
        doc_embs = model.encode(list(valid_texts), convert_to_numpy=True)
        
        # Compute cosine similarities
        query_norm = np.linalg.norm(query_emb)
        doc_norms = np.linalg.norm(doc_embs, axis=1)
        
        similarities = np.zeros(len(valid_texts))
        valid_mask = (doc_norms > 0) & (query_norm > 0)
        
        if np.any(valid_mask):
            similarities[valid_mask] = np.dot(doc_embs[valid_mask], query_emb) / (
                doc_norms[valid_mask] * query_norm
            )
        
        # Get top k results from valid documents
        top_valid_indices = np.argsort(similarities)[-k:][::-1]
        
        # Map back to original document indices
        result_docs = []
        for idx in top_valid_indices:
            if idx < len(valid_indices):
                orig_idx = valid_indices[idx]
                result_docs.append(docs[orig_idx])
        
        # Fill with remaining docs if needed
        while len(result_docs) < k and len(result_docs) < len(docs):
            for doc in docs:
                if doc not in result_docs:
                    result_docs.append(doc)
                    if len(result_docs) >= k:
                        break
        
        return result_docs[:k]
        
    except Exception as e:
        print(f"⚠️ Semantic retrieval error: {e}")
        return docs[:k]

def semantic_retrieve_with_scores(docs: List[str], 
                                  query: str, 
                                  k: int = 5,
                                  model_name: str = "all-MiniLM-L6-v2") -> Tuple[List[str], List[float]]:
    """Direct semantic retrieval using sentence transformers and return scores.
    
    Args:
        docs: List of documents to search
        query: Search query
        k: Number of documents to retrieve
        model_name: Name of the sentence transformer model
        
    Returns:
        Tuple of (retrieved documents, similarity scores)
        
    Raises:
        ValueError: If inputs are invalid
    """
    if not docs:
        raise ValueError("Documents list cannot be empty")
    
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    
    k = min(k, len(docs))
    
    try:
        model = SentenceTransformer(model_name)
        
        # Encode query and documents
        query_emb = model.encode([query], convert_to_numpy=True)[0]
        
        # Filter out empty documents
        valid_docs = [(i, doc) for i, doc in enumerate(docs) if doc and doc.strip()]
        
        if not valid_docs:
            print("⚠️ No valid documents for semantic search")
            return docs[:k], [0.0] * len(docs[:k])
        
        valid_indices, valid_texts = zip(*valid_docs)
        doc_embs = model.encode(list(valid_texts), convert_to_numpy=True)
        
        # Compute cosine similarities
        query_norm = np.linalg.norm(query_emb)
        doc_norms = np.linalg.norm(doc_embs, axis=1)
        
        similarities = np.zeros(len(valid_texts))
        valid_mask = (doc_norms > 0) & (query_norm > 0)
        
        if np.any(valid_mask):
            similarities[valid_mask] = np.dot(doc_embs[valid_mask], query_emb) / (
                doc_norms[valid_mask] * query_norm
            )
        
        # Get top k results from valid documents
        top_valid_indices = np.argsort(similarities)[-k:][::-1]
        
        # Map back to original document indices
        result_docs = []
        result_scores = []
        for idx in top_valid_indices:
            if idx < len(valid_indices):
                orig_idx = valid_indices[idx]
                result_docs.append(docs[orig_idx])
                result_scores.append(similarities[idx])
        
        # Fill with remaining docs if needed
        while len(result_docs) < k and len(result_docs) < len(docs):
            for doc in docs:
                if doc not in result_docs:
                    result_docs.append(doc)
                    result_scores.append(0.0) # Assign a default score for remaining docs
                    if len(result_docs) >= k:
                        break
        
        return result_docs[:k], result_scores[:k]
        
    except Exception as e:
        print(f"⚠️ Semantic retrieval error with scores: {e}")
        return docs[:k], [0.0] * len(docs[:k])

def retrieve_vqvae(vqvae: VQVAE,
                  codes: np.ndarray,
                  docs: List[str],
                  q_vec: np.ndarray,
                  k: int = 5) -> Tuple[List[str], np.ndarray]:
    """BEST PRACTICE: Encode both query and docs to 96-dim learned space for comparison.
    
    Args:
        vqvae: Trained VQ-VAE model
        codes: Array of document codes (not used in this approach)
        docs: List of document texts
        q_vec: Query vector (original 384-dim space)
        k: Number of documents to retrieve
        
    Returns:
        Tuple of (retrieved documents, similarity scores)
        
    Raises:
        ValueError: If inputs are invalid
    """
    if len(codes) != len(docs):
        raise ValueError(f"Codes length {len(codes)} doesn't match docs length {len(docs)}")
    
    if q_vec.size == 0:
        raise ValueError("Query vector cannot be empty")
    
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    
    k = min(k, len(docs))
    
    try:
        vqvae.eval()
        
        # Step 1: Encode query from 384-dim → 96-dim learned space
        with torch.no_grad():
            q_tensor = torch.from_numpy(q_vec).float()
            if q_tensor.dim() == 1:
                q_tensor = q_tensor.unsqueeze(0)
            
            query_encoded = vqvae.encoder(q_tensor).squeeze(0).cpu().numpy()  # 384→96
        
        # Step 2: We need the original 384-dim document embeddings to encode them
        # For now, let's re-embed the documents (this should be optimized by storing original embeddings)
        from . import embed_texts
        doc_embeddings = embed_texts(docs)  # Get 384-dim embeddings for all docs
        
        # Step 3: Encode ALL documents from 384-dim → 96-dim through VQ-VAE encoder
        doc_encoded_list = []
        with torch.no_grad():
            for doc_emb in doc_embeddings:
                doc_tensor = torch.from_numpy(doc_emb).float().unsqueeze(0)
                doc_encoded = vqvae.encoder(doc_tensor).squeeze(0).cpu().numpy()  # 384→96
                doc_encoded_list.append(doc_encoded)
        
        doc_encoded_array = np.array(doc_encoded_list)  # Shape: (n_docs, 96)
        
        # Step 4: Compare in 96-dim learned space (query vs all docs)
        query_norm = np.linalg.norm(query_encoded)
        doc_norms = np.linalg.norm(doc_encoded_array, axis=1)
        
        similarities = np.zeros(len(docs))
        valid_mask = (doc_norms > 1e-8) & (query_norm > 1e-8)
        
        if np.any(valid_mask):
            # Cosine similarity in 96-dim learned space
            similarities[valid_mask] = np.dot(doc_encoded_array[valid_mask], query_encoded) / (
                doc_norms[valid_mask] * query_norm
            )
        
        # Clip to valid cosine similarity range
        similarities = np.clip(similarities, -1.0, 1.0)
        
        # Get top k results
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        retrieved_docs = [docs[i] for i in top_indices]
        scores = similarities[top_indices]
        
        # Enhanced debug info
        print(f"🔍 VQ-VAE ENCODER-TO-ENCODER Debug:")
        print(f"   Query encoded norm: {query_norm:.4f}")
        print(f"   Doc encoded norms range: [{np.min(doc_norms):.4f}, {np.max(doc_norms):.4f}]")
        print(f"   Similarities range: [{np.min(similarities):.4f}, {np.max(similarities):.4f}]")
        print(f"   Valid docs: {np.sum(valid_mask)}/{len(docs)}")
        print(f"   Top 3 similarities: {scores[:3] if len(scores) >= 3 else scores}")
        
        return retrieved_docs, scores
        
    except Exception as e:
        print(f"⚠️ VQ-VAE retrieval error: {e}")
        import traceback
        traceback.print_exc()
        # Return simple fallback
        return docs[:k], np.zeros(k) 
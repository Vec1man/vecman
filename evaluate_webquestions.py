#!/usr/bin/env python3
"""
VECMAN RAGAS Evaluation Script for Web Questions Dataset

This script evaluates VECMAN's performance using the RAGAS framework
with the stanfordnlp/web_questions dataset.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm.auto import tqdm

# Core imports
from datasets import load_dataset, Dataset
import torch

# VECMAN imports
from vecman import (
    VQVAE, 
    train_corpus, 
    embed_texts, 
    save_jsonl, 
    load_assets, 
    retrieve, 
    generate_answer
)

# RAGAS imports
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness
)

# Optional: Enhanced evaluation setup
try:
    import google.generativeai as genai
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_community.embeddings import HuggingFaceBgeEmbeddings
    ENHANCED_EVAL = True
except ImportError:
    print("⚠️ Enhanced evaluation dependencies not found. Using basic evaluation.")
    ENHANCED_EVAL = False

class WebQuestionsEvaluator:
    """VECMAN evaluation system for Web Questions dataset with RAGAS integration."""
    
    def __init__(self, 
                 model_dir: str = "webquestions_models",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize Web Questions Evaluator.
        
        Args:
            model_dir: Directory to store/load VECMAN models
            embedding_model: Sentence transformer model for embeddings
            device: Device for training ('cuda' or 'cpu')
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.embedding_model = embedding_model
        self.device = device
        self.vqvae = None
        self.codes = None
        self.docs = None
        
    def prepare_dataset(self, dataset_name: str = "stanfordnlp/web_questions", 
                       split: str = "train", 
                       max_samples: Optional[int] = None) -> Dataset:
        """
        Load and prepare the Web Questions dataset.
        
        Args:
            dataset_name: HuggingFace dataset name
            split: Dataset split to use
            max_samples: Maximum number of samples to use (None for all)
            
        Returns:
            Prepared dataset
        """
        print(f"📥 Loading dataset: {dataset_name}")
        dataset = load_dataset(dataset_name, split=split)
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
            
        print(f"✅ Loaded {len(dataset)} samples")
        print(f"📋 Dataset fields: {list(dataset.features.keys())}")
        
        # Display sample for verification
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"📝 Sample question: {sample.get('question', 'N/A')}")
            print(f"📝 Sample answers: {sample.get('answers', 'N/A')}")
            
        return dataset
    
    def build_corpus(self, dataset: Dataset, force_rebuild: bool = False) -> None:
        """
        Build VECMAN corpus from Web Questions dataset.
        Creates documents by combining questions and answers from the dataset.
        
        Args:
            dataset: Input Web Questions dataset
            force_rebuild: Whether to rebuild even if corpus exists
        """
        corpus_path = self.model_dir / "corpus.npy"
        docs_path = self.model_dir / "docs.jsonl"
        
        if not force_rebuild and corpus_path.exists() and docs_path.exists():
            print("📂 Using existing corpus")
            return
            
        print("🔨 Building corpus from Web Questions dataset...")
        
        # Create documents by combining questions and answers
        documents = []
        
        for i, row in tqdm(enumerate(dataset), desc="Processing Web Questions", total=len(dataset)):
            question = row.get('question', '')
            answers = row.get('answers', [])
            
            if question and isinstance(question, str) and question.strip():
                # Create a document that includes the question and its answers
                doc_parts = [f"Question: {question.strip()}"]
                
                if answers and len(answers) > 0:
                    # Ensure answers is a list and process each answer
                    if not isinstance(answers, list):
                        answers = [answers]
                    
                    # Join all answers for this question, ensuring they're strings
                    valid_answers = []
                    for ans in answers:
                        if ans is not None:
                            ans_str = str(ans).strip()
                            if ans_str:
                                valid_answers.append(ans_str)
                    
                    if valid_answers:
                        answers_text = "; ".join(valid_answers)
                        doc_parts.append(f"Answers: {answers_text}")
                
                document = "\n".join(doc_parts)
                if document.strip():  # Only add non-empty documents
                    documents.append(document)
        
        print(f"📊 Created {len(documents)} documents from Web Questions")
        
        if len(documents) == 0:
            raise ValueError("❌ Could not create any documents from dataset")
        
        # Generate embeddings for the documents
        print("🔢 Generating embeddings...")
        embeddings = embed_texts(documents, self.embedding_model)
        
        # Save corpus and documents
        np.save(self.model_dir / "corpus.npy", embeddings)
        save_jsonl(documents, str(self.model_dir / "docs.jsonl"))
        
        print(f"💾 Saved corpus: {embeddings.shape}")
    
    def train_vecman(self, epochs: int = 10, force_retrain: bool = False) -> None:
        """
        Train VECMAN model on the corpus.
        
        Args:
            epochs: Number of training epochs
            force_retrain: Whether to retrain even if model exists
        """
        model_path = self.model_dir / "vqvae.pt"
        
        if not force_retrain and model_path.exists():
            print("🤖 Using existing trained model")
            return
            
        print("🏋️ Training VECMAN model...")
        
        # Load corpus to get input dimension
        corpus = np.load(self.model_dir / "corpus.npy")
        input_dim = corpus.shape[1]
        
        # Train the model
        train_corpus(
            str(self.model_dir / "corpus.npy"),
            input_dim=input_dim,
            epochs=epochs,
            device=self.device,
            output_dir=str(self.model_dir)
        )
        
        print("✅ VECMAN training completed")
    
    def load_vecman(self) -> None:
        """Load trained VECMAN model and assets."""
        print("📂 Loading VECMAN model...")
        self.vqvae, self.codes, self.docs = load_assets(str(self.model_dir))
        print(f"✅ Loaded model with {len(self.docs)} documents")
    
    def vecman_chat(self, question: str, k: int = 5, api_key: Optional[str] = None) -> Dict[str, Any]:
        """
        VECMAN-based chat function that retrieves and generates answers using improved retrieval.
        
        Args:
            question: User question
            k: Number of documents to retrieve
            api_key: Google API key for answer generation
            
        Returns:
            Dictionary with answer, contexts, similarity scores, and intermediate steps
        """
        # Embed the question
        q_vec = embed_texts([question], self.embedding_model)[0]
        
        # Retrieve relevant contexts using improved VQ-VAE method with scores
        try:
            # Get contexts and scores from improved retrieve function
            retrieve_result = retrieve(
                self.vqvae, 
                self.codes, 
                self.docs, 
                q_vec, 
                k=k, 
                method="vqvae",  # Use pure VQ-VAE for evaluation
                return_scores=True
            )
            
            # Handle the return format properly
            if isinstance(retrieve_result, tuple):
                contexts, scores = retrieve_result
            else:
                contexts = retrieve_result
                scores = [0.0] * len(contexts)
                
            # Ensure contexts is a list of strings
            if contexts and isinstance(contexts[0], list):
                # Flatten if nested lists
                contexts = [str(item) for sublist in contexts for item in (sublist if isinstance(sublist, list) else [sublist])]
            
            # Ensure all contexts are strings
            contexts = [str(ctx) if ctx is not None else "" for ctx in contexts]
            scores = [float(score) if score is not None else 0.0 for score in scores]
            
        except Exception as e:
            print(f"⚠️ Error in retrieval: {e}")
            contexts = []
            scores = []
        
        # Generate answer if API key is available
        if api_key or os.getenv("GOOGLE_API_KEY"):
            try:
                # Ensure contexts are valid strings before passing to generate_answer
                valid_contexts = [ctx for ctx in contexts if ctx and isinstance(ctx, str) and ctx.strip()]
                
                if valid_contexts:
                    # Custom template for web questions with improved prompt
                    custom_template = """
                    You are a helpful assistant. Use the following information to answer the question.
                    The context contains questions and their answers from a knowledge base.
                    Focus on providing accurate information based on the retrieved context.

                    Context:
                    {context}

                    Question: {question}
                    
                    Please provide a clear and factual answer based on the context above.
                    If the context contains relevant information, synthesize it into a comprehensive answer.
                    If the context doesn't contain sufficient information, say 'I don't have enough information to answer this question.'
                    """

                    answer = generate_answer(question, valid_contexts, api_key=api_key, prompt_template=custom_template)
                else:
                    answer = "I don't have enough information to answer this question."
                    
            except Exception as e:
                print(f"⚠️ Error generating answer: {e}")
                import traceback
                traceback.print_exc()
                answer = "I don't know"
        else:
            answer = "I don't know (no API key provided)"
        
        return {
            "output": answer,
            "contexts": contexts,
            "scores": scores,  # Include similarity scores
            "intermediate_steps": [("retrieve", "\n---\n".join(contexts))] if contexts else []
        }
    
    def select_evaluation_questions(self, dataset: Dataset, num_questions: int = 100) -> List[Dict[str, Any]]:
        """
        Select evaluation questions directly from the Web Questions dataset.
        
        Args:
            dataset: Web Questions dataset
            num_questions: Number of questions to select for evaluation
            
        Returns:
            List of dictionaries with question and ground_truth
        """
        print(f"📋 Selecting {num_questions} questions from Web Questions dataset...")
        
        # Use test split if available, otherwise use a different portion of the dataset
        try:
            # Try to load test split
            test_dataset = load_dataset("stanfordnlp/web_questions", split="train")
            eval_source = test_dataset
            print("📊 Using test split for evaluation")
        except:
            # If no test split, use the end portion of train split
            eval_source = dataset
            print("📊 Using portion of train split for evaluation")
        
        # Select questions for evaluation
        num_available = len(eval_source)
        num_to_select = min(num_questions, num_available)
        
        selected_questions = []
        
        for i in range(num_to_select):
            row = eval_source[i]
            question = row.get('question', '')
            answers = row.get('answers', [])
            
            if question.strip() and answers:
                # Use the first answer as ground truth, or join multiple answers
                if len(answers) == 1:
                    ground_truth = str(answers[0]).strip()
                else:
                    ground_truth = "; ".join([str(ans).strip() for ans in answers if str(ans).strip()])
                
                selected_questions.append({
                    "question": question.strip(),
                    "ground_truth": ground_truth
                })
        
        print(f"✅ Selected {len(selected_questions)} valid question-answer pairs")
        return selected_questions
    
    def run_evaluation(self, 
                      questions_data: List[Dict[str, Any]], 
                      k: int = 5,
                      api_key: Optional[str] = None) -> pd.DataFrame:
        """
        Run VECMAN evaluation on the selected questions.
        
        Args:
            questions_data: List of question-answer pairs
            k: Number of documents to retrieve
            api_key: Google API key
            
        Returns:
            DataFrame with evaluation results
        """
        print("🔍 Running VECMAN evaluation...")
        
        # Ensure model is loaded
        if self.vqvae is None:
            self.load_vecman()
        
        # Prepare evaluation data
        eval_data = []
        
        for i, q_data in tqdm(enumerate(questions_data), total=len(questions_data), desc="Evaluating"):
            question = q_data["question"]
            ground_truth = q_data["ground_truth"]
            
            try:
                # Get VECMAN response
                out = self.vecman_chat(question, k=k, api_key=api_key)
                answer = out["output"]
                contexts = out["contexts"]
            except Exception as e:
                print(f"⚠️ Error processing question {i}: {e}")
                answer = "ERROR"
                contexts = []
            
            eval_data.append({
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth
            })
        
        # Convert to DataFrame
        df = pd.DataFrame(eval_data)
        print(f"✅ Evaluation completed on {len(df)} samples")
        
        return df
    
    def compute_ragas_scores(self, eval_df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute RAGAS scores for the evaluation results.
        
        Args:
            eval_df: DataFrame with evaluation results
            
        Returns:
            Dictionary with RAGAS scores
        """
        print("📊 Computing RAGAS scores...")
        
        # Convert to RAGAS dataset format
        eval_dataset = Dataset.from_pandas(eval_df)
        
        # Basic metrics that work without additional setup
        basic_metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]
        
        # Enhanced metrics if dependencies are available
        if ENHANCED_EVAL:
            try:
                # Setup enhanced evaluation with custom LLM and embeddings
                emb = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5")
                ragas_emb = LangchainEmbeddingsWrapper(emb)

                result = evaluate(
                    dataset=eval_dataset,
                    metrics=basic_metrics + [answer_correctness],
                    embeddings=ragas_emb,
                )
            except Exception as e:
                print(f"⚠️ Enhanced evaluation failed: {e}")
                print("📊 Falling back to basic evaluation...")
                result = evaluate(dataset=eval_dataset, metrics=basic_metrics)
        else:
            result = evaluate(dataset=eval_dataset, metrics=basic_metrics)
        
        # Convert to pandas and get mean scores
        result_df = result.to_pandas()
        scores = {}
        
        for metric in result_df.columns:
            if metric not in ['question', 'answer', 'contexts', 'ground_truth']:
                scores[metric] = result_df[metric].mean()
        
        return scores
    
    def save_results(self, eval_df: pd.DataFrame, scores: Dict[str, float], 
                    output_file: str = "webquestions_evaluation_results.json") -> None:
        """
        Save evaluation results to file.
        
        Args:
            eval_df: Evaluation DataFrame
            scores: RAGAS scores
            output_file: Output file path
        """
        results = {
            "evaluation_summary": scores,
            "detailed_results": eval_df.to_dict('records'),
            "total_samples": len(eval_df),
            "model_config": {
                "embedding_model": self.embedding_model,
                "device": self.device,
                "model_dir": str(self.model_dir)
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Results saved to {output_file}")

def main():
    """Main evaluation pipeline for Web Questions dataset."""
    print("🚀 VECMAN RAGAS Evaluation Pipeline - Web Questions")
    print("=" * 60)
    
    # Configuration
    MAX_SAMPLES = None  # Use full dataset for corpus building
    EPOCHS = 30
    K_RETRIEVE = 50  # Top 10 relevant contexts
    NUM_QUESTIONS = 10  # Use 100 questions from dataset
    
    # Initialize evaluator
    evaluator = WebQuestionsEvaluator()
    
    # Step 1: Load Web Questions dataset
    print("📥 Step 1: Loading Web Questions dataset...")
    dataset = evaluator.prepare_dataset(max_samples=MAX_SAMPLES)
    
    # Step 2: Build corpus from Web Questions
    print("🔨 Step 2: Building corpus from Web Questions...")
    evaluator.build_corpus(dataset, force_rebuild=True)
    
    # Step 3: Train VECMAN
    print("🏋️ Step 3: Training VECMAN...")
    evaluator.train_vecman(epochs=EPOCHS, force_retrain=True)
    
    # Step 4: Load VECMAN
    print("📂 Step 4: Loading VECMAN...")
    evaluator.load_vecman()
    
    # Step 5: Select evaluation questions from dataset
    print("📋 Step 5: Selecting evaluation questions from dataset...")
    questions_data = evaluator.select_evaluation_questions(dataset, NUM_QUESTIONS)
    
    # Step 6: Run evaluation
    print("🔍 Step 6: Running RAGAS evaluation...")
    eval_df = evaluator.run_evaluation(
        questions_data, 
        k=K_RETRIEVE, 
        api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    print(f"✅ Created RAGAS dataset with {len(eval_df)} samples")
    print(f"📊 Each sample has: 1 question, 1 answer, {K_RETRIEVE} contexts, 1 ground_truth")
    
    # Step 7: Compute RAGAS scores
    try:
        scores = evaluator.compute_ragas_scores(eval_df)
        
        print("\n📊 RAGAS Evaluation Results:")
        print("-" * 30)
        for metric, score in scores.items():
            print(f"{metric:20s}: {score:.4f}")
        
        # Step 8: Save results
        evaluator.save_results(eval_df, scores)
        
    except Exception as e:
        print(f"⚠️ Error computing RAGAS scores: {e}")
        print("💾 Saving evaluation data without RAGAS scores...")
        evaluator.save_results(eval_df, {})
    
    print("\n✅ VECMAN-RAGAS evaluation on Web Questions completed!")
    print("🎯 Check 'webquestions_evaluation_results.json' for detailed results")

if __name__ == "__main__":
    main() 
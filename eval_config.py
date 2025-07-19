"""
VECMAN Evaluation Configuration

Customize evaluation parameters here.
"""

import os

# Dataset Configuration
DATASET_NAME = "aurelio-ai/ai-arxiv2-ragas-mixtral"
DATASET_SPLIT = "train"
MAX_SAMPLES = 500  # Set to None for full dataset (can be very large)

# Model Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Sentence transformer model
DEVICE = "cuda"  # "cuda" or "cpu"
MODEL_DIR = "vecman_models"

# Training Configuration
EPOCHS = 5
LATENT_BITS = 16  # VQ-VAE latent bits
BATCH_SIZE = 4096

# Retrieval Configuration
K_RETRIEVE = 5  # Number of documents to retrieve
RETRIEVAL_THRESHOLD = None  # Optional similarity threshold

# Generation Configuration
GENERATIVE_MODEL = "gemini-2.0-flash"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Set via environment variable

# Evaluation Configuration
RAGAS_METRICS = [
    "faithfulness",
    "answer_relevancy", 
    "context_precision",
    "context_recall",
    "answer_correctness"  # Requires enhanced dependencies
]

# Output Configuration
OUTPUT_FILE = "vecman_evaluation_results.json"
SAVE_DETAILED_RESULTS = True
SAVE_INDIVIDUAL_SCORES = True

# Performance Configuration
USE_MULTIPROCESSING = False  # Set to True for faster processing
NUM_WORKERS = 4  # Number of worker processes

# Debug Configuration
DEBUG_MODE = True
VERBOSE_LOGGING = True

# Advanced Configuration
FORCE_REBUILD_CORPUS = True
FORCE_RETRAIN_MODEL = True
CACHE_EMBEDDINGS = True

# Experiment Configuration
EXPERIMENT_NAME = "vecman_baseline"
EXPERIMENT_TAGS = ["vecman", "vqvae", "ragas", "arxiv"]

def get_config():
    """Get all configuration as a dictionary."""
    return {
        "dataset": {
            "name": DATASET_NAME,
            "split": DATASET_SPLIT,
            "max_samples": MAX_SAMPLES
        },
        "model": {
            "embedding_model": EMBEDDING_MODEL,
            "device": DEVICE,
            "model_dir": MODEL_DIR
        },
        "training": {
            "epochs": EPOCHS,
            "latent_bits": LATENT_BITS,
            "batch_size": BATCH_SIZE
        },
        "retrieval": {
            "k": K_RETRIEVE,
            "threshold": RETRIEVAL_THRESHOLD
        },
        "generation": {
            "model": GENERATIVE_MODEL,
            "api_key": GOOGLE_API_KEY
        },
        "evaluation": {
            "metrics": RAGAS_METRICS,
            "output_file": OUTPUT_FILE
        },
        "experiment": {
            "name": EXPERIMENT_NAME,
            "tags": EXPERIMENT_TAGS
        }
    }

def print_config():
    """Print current configuration."""
    config = get_config()
    print("🔧 VECMAN Evaluation Configuration")
    print("=" * 40)
    
    for section, params in config.items():
        print(f"\n{section.upper()}:")
        for key, value in params.items():
            if key == "api_key" and value:
                value = "***" + value[-4:] if len(value) > 4 else "***"
            print(f"  {key:20s}: {value}")

if __name__ == "__main__":
    print_config() 
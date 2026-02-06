"""
Entry point for running HyP-DLM.

Similar to run.py but uses the new HyP-DLM architecture.
"""

import argparse
import json
import os
import warnings
from datetime import datetime

import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from src.HyPDLM import HyPDLM, HyPDLMConfig
from src.utils import LLM_Model, setup_logging
from src.evaluate import Evaluator

# Load environment variables from .env file
load_dotenv()

# Auto-detect device: MPS for M1 Mac, CUDA for Nvidia, CPU fallback
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

warnings.filterwarnings('ignore')


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run HyP-DLM retrieval and QA")
    
    # Model configs
    parser.add_argument("--spacy_model", type=str, default="en_core_web_trf",
                        help="SpaCy model for NER")
    parser.add_argument("--embedding_model", type=str, default="sentence-transformers/all-mpnet-base-v2",
                        help="Path to embedding model")
    parser.add_argument("--llm_model", type=str, default="z-ai/glm-4.7-flash",
                        help="LLM model for proposition extraction and QA")
    
    # Dataset
    parser.add_argument("--dataset_name", type=str, default="2wikimultihop",
                        help="Dataset name")
    
    # HyP-DLM specific configs
    parser.add_argument("--num_clusters", type=int, default=100,
                        help="Number of clusters for semantic masking")
    parser.add_argument("--top_p_clusters", type=int, default=10,
                        help="Top P clusters to consider")
    parser.add_argument("--max_hops", type=int, default=5,
                        help="Maximum propagation hops")
    parser.add_argument("--propagation_factor", type=float, default=0.3,
                        help="Beta factor for A_sem alias propagation")
    parser.add_argument("--affinity_threshold", type=float, default=0.7,
                        help="Threshold for A_sem similarity")
    
    # Processing
    parser.add_argument("--max_workers", type=int, default=16,
                        help="Max parallel workers")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for embeddings")
    
    # Mode
    parser.add_argument("--skip_indexing", action="store_true",
                        help="Skip indexing if index exists")
    
    return parser.parse_args()


def load_dataset(dataset_name):
    """Load dataset from disk."""
    questions_path = f"dataset/{dataset_name}/questions.json"
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    
    chunks_path = f"dataset/{dataset_name}/chunks.json"
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    passages = [f'{idx}:{chunk}' for idx, chunk in enumerate(chunks)]
    return questions, passages


def load_embedding_model(embedding_model_path):
    """Load sentence transformer model."""
    print(f"Loading embedding model on device: {DEVICE}")
    return SentenceTransformer(embedding_model_path, device=DEVICE)


def main():
    # Setup
    time = datetime.now()
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    args = parse_arguments()
    
    # Create results directory
    results_dir = f"results/{args.dataset_name}/{time_str}"
    os.makedirs(results_dir, exist_ok=True)
    setup_logging(f"{results_dir}/log.txt")
    
    # Load models
    print("Loading models...")
    embedding_model = load_embedding_model(args.embedding_model)
    llm_model = LLM_Model(args.llm_model)
    
    # Load dataset
    print(f"Loading dataset: {args.dataset_name}")
    questions, passages = load_dataset(args.dataset_name)
    
    # Create config
    config = HyPDLMConfig(
        embedding_model=embedding_model,
        llm_model=llm_model,
        spacy_model=args.spacy_model,
        working_dir=f"./hypdlm_index/{args.dataset_name}",
        num_clusters=args.num_clusters,
        top_p_clusters=args.top_p_clusters,
        max_hops=args.max_hops,
        propagation_factor=args.propagation_factor,
        affinity_threshold=args.affinity_threshold,
        max_workers=args.max_workers,
        batch_size=args.batch_size
    )
    
    # Initialize HyP-DLM
    print("Initializing HyP-DLM...")
    hypdlm = HyPDLM(config)
    
    # Index or load
    index_exists = os.path.exists(os.path.join(config.working_dir, "hypergraph"))
    
    if args.skip_indexing and index_exists:
        print("Loading existing index...")
        hypdlm.load_index()
    else:
        print("Building index...")
        hypdlm.index(passages)
    
    # Run QA
    print(f"Running QA on {len(questions)} questions...")
    results = hypdlm.qa(questions)
    
    # Save results
    output_path = f"{results_dir}/predictions.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Results saved to {output_path}")
    
    # Evaluate
    print("Evaluating...")
    evaluator = Evaluator(
        llm_model=llm_model,
        predictions_path=output_path
    )
    evaluator.evaluate(max_workers=args.max_workers)
    
    print("Done!")


if __name__ == "__main__":
    main()

import os
import json
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from src.HyPDLM import HyPDLM, HyPDLMConfig
from src.utils import LLM_Model

# Load environment variables from .env file
load_dotenv()

# Auto-detect device
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# Initialize FastAPI app
app = FastAPI(
    title="HyP-DLM API",
    description="Hypergraph Propagation with Dynamic Logic Modulation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
hypdlm_instance: Optional[HyPDLM] = None


# Request/Response models
class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    evidence: List[str]
    source_chunks: List[str]
    num_hops: int


class BatchQueryRequest(BaseModel):
    questions: List[str]


class BatchQueryResponse(BaseModel):
    results: List[QueryResponse]


class HealthResponse(BaseModel):
    status: str
    indexed: bool
    num_entities: int
    num_propositions: int


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    global hypdlm_instance
    
    # Get config from environment or use defaults
    embedding_model_path = os.getenv("EMBEDDING_MODEL", "model/all-mpnet-base-v2")
    llm_model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
    spacy_model = os.getenv("SPACY_MODEL", "en_core_web_trf")
    dataset_name = os.getenv("DATASET_NAME", "2wikimultihop")
    working_dir = os.getenv("WORKING_DIR", f"./hypdlm_index/{dataset_name}")
    
    print(f"Loading models on device: {DEVICE}...")
    embedding_model = SentenceTransformer(embedding_model_path, device=DEVICE)
    llm_model = LLM_Model(llm_model_name)
    
    config = HyPDLMConfig(
        embedding_model=embedding_model,
        llm_model=llm_model,
        spacy_model=spacy_model,
        working_dir=working_dir
    )
    
    hypdlm_instance = HyPDLM(config)
    
    # Load existing index if available
    if os.path.exists(os.path.join(working_dir, "hypergraph")):
        print("Loading existing index...")
        hypdlm_instance.load_index()
        print("Index loaded successfully!")
    else:
        print("No existing index found. Use POST /index to create one.")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if hypdlm_instance is None:
        return HealthResponse(
            status="initializing",
            indexed=False,
            num_entities=0,
            num_propositions=0
        )
    
    indexed = hypdlm_instance.hypergraph is not None
    num_entities = hypdlm_instance.hypergraph.num_entities if indexed else 0
    num_props = hypdlm_instance.hypergraph.num_propositions if indexed else 0
    
    return HealthResponse(
        status="healthy",
        indexed=indexed,
        num_entities=num_entities,
        num_propositions=num_props
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Process a single query."""
    if hypdlm_instance is None or hypdlm_instance.hypergraph is None:
        raise HTTPException(status_code=503, detail="Index not loaded")
    
    results = hypdlm_instance.qa([{"question": request.question}])
    result = results[0]
    
    return QueryResponse(
        question=result["question"],
        answer=result.get("answer", ""),
        evidence=result.get("evidence", []),
        source_chunks=result.get("source_chunks", []),
        num_hops=result.get("num_hops", 0)
    )


@app.post("/batch_query", response_model=BatchQueryResponse)
async def batch_query(request: BatchQueryRequest):
    """Process multiple queries at once."""
    if hypdlm_instance is None or hypdlm_instance.hypergraph is None:
        raise HTTPException(status_code=503, detail="Index not loaded")
    
    questions = [{"question": q} for q in request.questions]
    results = hypdlm_instance.qa(questions)
    
    responses = []
    for result in results:
        responses.append(QueryResponse(
            question=result["question"],
            answer=result.get("answer", ""),
            evidence=result.get("evidence", []),
            source_chunks=result.get("source_chunks", []),
            num_hops=result.get("num_hops", 0)
        ))
    
    return BatchQueryResponse(results=responses)


class IndexRequest(BaseModel):
    dataset_name: str


@app.post("/index")
async def index_dataset(request: IndexRequest):
    """Index a dataset."""
    if hypdlm_instance is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        # Load dataset
        chunks_path = f"dataset/{request.dataset_name}/chunks.json"
        if not os.path.exists(chunks_path):
            raise HTTPException(status_code=404, detail=f"Dataset not found: {request.dataset_name}")
        
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        
        passages = [f'{idx}:{chunk}' for idx, chunk in enumerate(chunks)]
        
        # Index
        hypdlm_instance.index(passages)
        
        return {
            "status": "success",
            "message": f"Indexed {len(passages)} passages",
            "num_entities": hypdlm_instance.hypergraph.num_entities,
            "num_propositions": hypdlm_instance.hypergraph.num_propositions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

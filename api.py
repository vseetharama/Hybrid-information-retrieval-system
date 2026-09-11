from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import DEFAULT_TOP_K, GROUND_TRUTH_FILE
from evaluator import evaluate, load_ground_truth
from indexer import build_index
from search_engine import SearchEngine


app = FastAPI(title="Hybrid Information Retrieval System", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
engine = SearchEngine()
frontend_dir = Path(__file__).resolve().parent / "frontend"
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=DEFAULT_TOP_K, gt=0)


class EvaluateRequest(BaseModel):
    k: int = Field(default=DEFAULT_TOP_K, gt=0)


@app.get("/")
def frontend() -> FileResponse:
    return FileResponse(frontend_dir / "index.html")


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "indexed": engine._loaded}


@app.post("/index")
def index() -> dict:
    try:
        result = build_index()
        engine._loaded = False
        return {"status": "indexed", **result}
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/search")
def search(request: SearchRequest) -> dict:
    try:
        return engine.search(request.query, request.top_k)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/evaluate")
def evaluation(request: EvaluateRequest) -> dict:
    try:
        return evaluate(load_ground_truth(GROUND_TRUTH_FILE), engine.search, request.k)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

"""FastAPI backend exposing the movie search engine to the React frontend."""
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from indexer import MovieSearchEngine
from preprocessing import query_terms

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "movies.json"

app = FastAPI(title="Movie Search Engine API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

engine = MovieSearchEngine(DATA_PATH)


class Movie(BaseModel):
    id: int
    title: str
    year: int
    director: str
    genres: list[str]
    cast: str
    plot: str
    score: float


class SearchResponse(BaseModel):
    query: str
    method: str
    terms: list[str]
    results: list[Movie]


class StatsResponse(BaseModel):
    movie_count: int
    vocabulary_size: int


@app.get("/api/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    return StatsResponse(
        movie_count=len(engine.movies),
        vocabulary_size=engine.vocabulary_size(),
    )


@app.get("/api/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1, description="Search query"),
    method: Literal["bm25", "tfidf", "hybrid"] = "bm25",
    top_k: int = Query(10, ge=1, le=50),
    alpha: float = Query(0.5, ge=0.0, le=1.0),
) -> SearchResponse:
    results = engine.search(q, method=method, top_k=top_k, alpha=alpha)
    return SearchResponse(
        query=q,
        method=method,
        terms=query_terms(q),
        results=[Movie(**r) for r in results],
    )

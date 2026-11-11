
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag_engine import (
    extract_pdf_pages,
    split_into_chunks,
    build_retriever,
    retrieve_chunks,
    generate_with_gemini,
)

PDF_PATH = Path(__file__).parent / "KnowledgePilot_Test_Document.pdf"

chunks = None
retriever = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global chunks, retriever

    if not PDF_PATH.exists():
        raise RuntimeError(f"Test PDF not found: {PDF_PATH}")

    pdf_bytes = PDF_PATH.read_bytes()
    pages = extract_pdf_pages(pdf_bytes)
    chunks = split_into_chunks(pages, PDF_PATH.name)
    retriever = build_retriever(chunks)

    print(f"KnowledgePilot adapter ready: {len(chunks)} chunks")
    yield


app = FastAPI(title="KnowledgePilot Agent Adapter", lifespan=lifespan)


class AgentRequest(BaseModel):
    input: str


@app.get("/health")
def health():
    return {"status": "ready", "agent": "KnowledgePilot"}


@app.post("/agent")
def run_agent(request: AgentRequest, retrieval_only: bool = False):
    if not request.input.strip():
        raise HTTPException(status_code=400, detail="Input cannot be empty")

    results = retrieve_chunks(
        request.input,
        chunks,
        retriever,
        top_k=5,
    )

    if not results:
        return {
            "output": "I could not find relevant information in the document.",
            "tool_calls": [],
            "metadata": {"retrieved_chunks": 0},
        }
    if retrieval_only:
        return {
            "output": "\n\n".join(
                item.get("text", "") for item in results
                if isinstance(item, dict)
            ),
            "tool_calls": [
                {
                    "tool": "document_retrieval",
                    "retrieved_chunks": len(results),
                }
            ],
            "metadata": {
                "retrieved_chunks": len(results),
                "mode": "retrieval_only",
                "sources": [
                    {
                        "source": item.get("source"),
                        "page": item.get("page"),
                        "score": item.get("score"),
                    }
                    for item in results
                    if isinstance(item, dict)
                ],
            },
        }

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY environment variable is not configured",
        )

    answer = generate_with_gemini(request.input, results, api_key)

    return {
        "output": answer,
        "tool_calls": [
            {
                "tool": "document_retrieval",
                "retrieved_chunks": len(results),
            }
        ],
        "metadata": {
            "retrieved_chunks": len(results),
        },
    }

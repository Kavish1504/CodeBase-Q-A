from __future__ import annotations
from loguru import logger
from ingestion.embedder import ingest_repo,load_vectorstore
from generation.qa_chain import CodebaseQA
from retrieval.retriever import build_hybrid_retriever
from pydantic import BaseModel
from fastapi import FastAPI,BackgroundTasks,HTTPException
from fastapi.middleware.cors import CORSMiddleware

_qa_registry:dict[str,CodebaseQA]={}
_indexing_status:dict[str,str]={}
class IngestRequest(BaseModel):
    repo_url:str
    force_reindex:bool=False


class AskRequest(BaseModel):
    repo_url:str
    question:str

class AskResponse(BaseModel):
    answer:str
    sources:list[dict[str,str]]

app=FastAPI(
    title="Codebase Q&A API",
    description="Ask natural-language questions about any GitHub repository.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def _do_ingest(repo_url:str,force_reindex:bool=False):
    try:
        _indexing_status[repo_url]="pending"
        vs=ingest_repo(repo_url,force_reindex=force_reindex)
        retriever=build_hybrid_retriever(vs)
        _qa_registry[repo_url]=CodebaseQA(retriever)
        _indexing_status[repo_url]="done"
        logger.info(f"Ingestion complete for {repo_url}")
    except Exception as e:
        _indexing_status[repo_url] = f"error:{e}"
        logger.error(f"Ingestion failed for {repo_url}: {e}")


@app.get("/health")
def health()->dict:
    return {"status":"ok"}

@app.post("/ingest",status_code=202)
def start_ingest(req:IngestRequest,background_tasks:BackgroundTasks)->dict:
    background_tasks.add_task(_do_ingest,req.repo_url,req.force_reindex)
    _indexing_status[req.repo_url]="pending"
    return {"message": "Ingestion started", "repo_url": req.repo_url}

@app.get("/status")
def get_status(repo_url:str)->dict:
    status=_indexing_status.get(repo_url,"not_started")
    return {"repo_url":repo_url,"status":status}

@app.post("/ask",response_model=AskResponse)
def ask_question(req:AskRequest)->AskResponse:
    try:
        vs=load_vectorstore(req.repo_url)
        retriever=build_hybrid_retriever(vs)
        qa=CodebaseQA(retriever)
        _qa_registry[req.repo_url]=qa
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Repo '{req.repo_url}' not indexed yet. Call POST /ingest first.",
        )
    
    result=qa.ask(req.question)
    return AskResponse(answer=result.answer,sources=result.sources)


@app.delete("/reset")
def reset_memory(repo_url:str)->dict:
    qa=_qa_registry.get(repo_url)
    if qa:
        qa.reset_memory()
        return {"message":"memory cleared"}
    return {"message":"No Active Session Found"}





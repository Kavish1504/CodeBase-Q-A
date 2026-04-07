import hashlib
from ingestion.code_chunker import chunk_all_files
from ingestion.repo_loader import load_repo
from loguru import logger
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from tqdm import tqdm
from config import settings
from pathlib import Path

_COLLECTION_PREFIX="codebase_"
_BATCH_SIZE=100

def _collection_name(repo_url:str)->str:
    repo_name=repo_url.rstrip("/").split("/")[-1].removesuffix(".git")
    hash=hashlib.md5(repo_name.encode()).hexdigest()[:6]
    return f"{_COLLECTION_PREFIX}{repo_name}_{hash}"[:63]

def _get_embeddings()->HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device":"cpu"},
        encode_kwargs={"normalize_embeddings":True},
    )

def ingest_repo(repo_url:str,force_reindex:bool=False)->Chroma:
    collection=_collection_name(repo_url)
    embeddings=_get_embeddings()
    persist_dir=settings.chroma_persist_dir
    if not force_reindex and Path(persist_dir).exists():
        existing=Chroma(
            collection_name=collection,
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
        count=existing._collection.count()
        if count>0:
            logger.info(
                f"Collection '{collection}' already has {count} chunks. "
                "Skipping re-index. Pass force_reindex=True to rebuild."
            )
            return existing
        
    logger.info("Step 1/3 — Loading repository…")
    files=load_repo(repo_url,settings.repo_clone_dir)

    logger.info("Step 2/3 — Chunking source files…")
    docs:list[Document]=chunk_all_files(files)

    if not docs:
        raise ValueError("No documents produced — check repo URL and file types.")
    
    logger.info(f"Step 3/3 — Embedding {len(docs)} chunks into '{collection}'…")
    vectorstore: Chroma|None=None

    for i in tqdm(range(0,len(docs),_BATCH_SIZE),desc="Embeddings"):
        batch=docs[i:i+_BATCH_SIZE]
        if vectorstore is None:
            vectorstore=Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=persist_dir,
                collection_name=collection
            )
        else:
            vectorstore.add_documents(batch)

    logger.info(f"✅ Indexed {len(docs)} chunks from {len(files)} files.")
    return vectorstore

def load_vectorstore(repo_url:str)->Chroma:
    collection=_collection_name(repo_url)
    embeddings=_get_embeddings()
    vs=Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir
    )
    count = vs._collection.count()
    if count == 0:
        raise FileNotFoundError(
            f"No indexed data found for '{repo_url}'. Run ingest_repo() first."
        )
    logger.info(f"Loaded vectorstore '{collection}' with {count} chunks.")
    return vs

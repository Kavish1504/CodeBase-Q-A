from langchain.schema import BaseRetriever, Document
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from loguru import logger
from config import settings
def rrf_score(rank:int,k:int):
    return 1.0/(rank+k)

def build_hybrid_retriever(vectorstore:Chroma, k:int |None=None, all_docs:list[Document] |None=None)->EnsembleRetriever:
    k=k or settings.retriever_k
    semantic_retriever=vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k":k,"fetch_k":k*3},
    )
    if all_docs:
        bm25_retriever=BM25Retriever.from_documents(all_docs)
        bm25_retriever.k=k
        retriever=EnsembleRetriever(
            retrievers=[semantic_retriever, bm25_retriever],
            weights=[0.6, 0.4],
        )
        logger.info("Hybrid retriever (semantic + BM25) ready.")
    else:
        retriever=semantic_retriever
        logger.info("Semantic-only retriever ready (no BM25 docs supplied).")
    
    return retriever

def rerank_with_cohere(query:str,docs:list[Document],top_n:int =5 ):
    if not settings.cohere_api_key:
        logger.debug("No Cohere key — skipping rerank.")
        return docs[:top_n]
    
    try:
        import cohere
        co=cohere.Client(settings.cohere_api_key)
        texts=[d.page_content for d in docs]
        response = co.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=texts,
            top_n=top_n,
        )
        reranked=[docs[r.index] for r in response.results]
        logger.info(f"Cohere reranked {len(docs)} → {len(reranked)} docs.")
        return reranked
    except Exception as e:
        return docs[:top_n]


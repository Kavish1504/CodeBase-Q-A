from __future__ import annotations
from typing import Any
from langchain.schema import BaseRetriever,Document
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from retrieval.retriever import rerank_with_cohere
from generation.prompt_templates import CONDENSE_QUESTION_PROMPT,CODE_QA_PROMPT
from dataclasses import dataclass,field
from config import settings
from loguru import logger

@dataclass
class QAResult:
    answer:str
    sources:list[dict[str,str]]
    raw_docs:list[Document] = field(default_factory=list)


def _format_sources(docs: list[Document])->list[dict[str,str]]:
    seen: set[str]=set()
    sources:list[dict[str,str]]=[]
    for doc in docs:
        meta=doc.metadata
        key=meta.get("source",meta.get("file_path","unknown"))
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "file":meta.get("file_path","unknown"),
                "lines": f"{meta.get('start_line', '?')}–{meta.get('end_line', '?')}",
                "language":meta.get("language",""),
                "source":key,
            }
        )
    return sources


class CodebaseQA:

    def __init__(
            self,
            retriever:BaseRetriever,
            memory_window:int=5,
            use_reranker:bool=True,
            temperature:float=0.0,
            model="llama-3.3-70b-versatile")->None:
        
        self.retriever=retriever
        self.use_reranker=use_reranker
        self._llm=ChatGroq(
            model=model,
            temperature=temperature,
            groq_api_key=settings.groq_api_key
        )

        self._memory=ConversationBufferWindowMemory(
            k=memory_window,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

        self._chain = ConversationalRetrievalChain.from_llm(
            llm=self._llm,
            retriever=self.retriever,
            memory=self._memory,
            combine_docs_chain_kwargs={"prompt": CODE_QA_PROMPT},
            condense_question_llm=self._llm,
            return_source_documents=True,
            verbose=False,
        )

    def ask(self,question:str)->QAResult:
        logger.info(f"Q: {question}")
        result:dict[str,Any]=self._chain.invoke({"question":question})
        raw_docs:list[Document]=result.get("source_documents",[])
        if self.use_reranker and raw_docs:
            raw_docs=rerank_with_cohere(question,raw_docs,top_n=5)

        answer:str=result.get("answer","")
        sources=_format_sources(raw_docs)
        logger.info(f"A: {answer[:120]}…")
        return QAResult(answer,sources,raw_docs)
    
    def clear_memory(self):
        self._memory.clear()
        logger.info("Conversation memory cleared.")

def build_qa(retriever:BaseRetriever,**kwargs:Any)->CodebaseQA:
    return CodebaseQA(retriever,**kwargs)





    









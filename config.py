from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    groq_api_key:str=""
    embedding_model:str="BAAI/bge-base-en-v1.5"
    chroma_persist_dir:str="./chroma_db"
    repo_clone_dir:str="./cloned_repos"
    cohere_api_key:str=""
    github_token:str=""
    max_chunk_size:int=1500
    chunk_overlap:int=200
    retriever_k:int = 6
    log_level:str="INFO"
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"       
    }

settings=Settings()
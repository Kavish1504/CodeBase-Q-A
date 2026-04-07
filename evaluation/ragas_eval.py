from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
from typing import Optional

from datasets import Dataset
from loguru import logger
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    faithfulness,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from config import settings
from generation.qa_chain import CodebaseQA

DEFAULT_TEST_QUESTIONS = [
    "How are commands defined in Click?",
    "What does click.Context do?",
    "How are options implemented in Click?",
]


def build_ragas_dataset(
    qa: CodebaseQA,
    questions: list[str],
    ground_truths: Optional[list[str]] = None,
) -> Dataset:
    data: dict[str, list] = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    gt = ground_truths or [""] * len(questions)

    for question, truth in zip(questions, gt):
        result = qa.ask(question)
        data["question"].append(question)
        data["answer"].append(result.answer)
        data["contexts"].append([doc.page_content[:500] for doc in result.raw_docs[:3]])
        data["ground_truth"].append(truth)
        qa.clear_memory()

    return Dataset.from_dict(data)


def run_evaluation(
    qa: CodebaseQA,
    questions: list[str] | None = None,
    ground_truths: list[str] | None = None,
    output_path: str = "./evaluation/results.json",
) -> dict:
    questions = questions or DEFAULT_TEST_QUESTIONS
    logger.info(f"Running RAGAS evaluation on {len(questions)} questions...")

    groq_llm = LangchainLLMWrapper(
        ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            groq_api_key=settings.groq_api_key,
        )
    )

    hf_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    )

    dataset = build_ragas_dataset(qa, questions, ground_truths)
    metrics = [faithfulness, answer_relevancy, context_precision]

    scores = evaluate(
        dataset,
        metrics=metrics,
        llm=groq_llm,                  
        embeddings=hf_embeddings,      
    )

    df = scores.to_pandas()

    numeric_df = df.select_dtypes(include=["number"])

    scores_dict = numeric_df.mean().to_dict()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(scores_dict, f, indent=2)

    logger.info(f"RAGAS scores: {scores_dict}")
    logger.info(f"Results saved to {output_path}")
    return scores_dict


if __name__ == "__main__":
    import argparse
    from ingestion.embedder import load_vectorstore
    from retrieval.retriever import build_hybrid_retriever

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_url", required=True)
    args = parser.parse_args()

    vs = load_vectorstore(args.repo_url)
    retriever = build_hybrid_retriever(vs)
    qa = CodebaseQA(retriever)
    run_evaluation(qa)
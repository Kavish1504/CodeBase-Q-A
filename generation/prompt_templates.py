from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

_SYSTEM = """You are an expert software engineer and code reviewer.
You have been given access to snippets from a codebase.

RULES:
- Base your answer ONLY on the retrieved code snippets below.
- Always cite the exact file path and line range (e.g. auth/jwt.py:34-67).
- If multiple files are relevant, mention all of them.
- If the answer is not in the provided snippets, say:
  I couldn't find this in the indexed portions of the codebase.
  Do NOT hallucinate code or file names.
- Keep explanations concise but complete.
"""

CODE_QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(_SYSTEM),
        HumanMessagePromptTemplate.from_template(
            "Retrieved code snippets:\n{context}\n\nQuestion: {question}\n\nAnswer (include file paths and line numbers):"
        ),
    ]
)

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
    """Given the following conversation history and a follow-up question,
rephrase the follow-up as a standalone question. Preserve technical terms exactly.

Chat history:
{chat_history}

Follow-up question: {question}
Standalone question:"""
)
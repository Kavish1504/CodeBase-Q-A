from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loguru import logger
import streamlit as st
from generation.qa_chain import CodebaseQA,QAResult
from retrieval.retriever import build_hybrid_retriever
from ingestion.embedder import ingest_repo,load_vectorstore

st.set_page_config(
    page_title="Codebase Q&A",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
<style>
    .source-card {
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 4px 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        color: #cdd6f4;
    }
    .source-card .file { color: #89b4fa; font-weight: 600; }
    .source-card .lines { color: #a6e3a1; }
    .source-card .lang { color: #f38ba8; }
    .stChatMessage { border-radius: 12px; }
</style>
""",
    unsafe_allow_html=True,
)

def _getqa()->CodebaseQA|None:
    return st.session_state.get("qa_instance")

def _get_messages()->list[dict]:
    if "messages" not in st.session_state:
        st.session_state.messages=[]
    return st.session_state.messages


with st.sidebar:
    st.title("CodeBase QA")
    st.caption("Ask anything about any GitHub repo.")
    st.divider()
    st.subheader("📦 Index a Repository")

    repo_url=st.text_input(
        "Github URL",
        placeholder="https://github.com/tiangolo/fastapi",
        key="repo_url_input",
    )
    force_reindex=st.checkbox("Force-Reindex",value=False)
    col1,col2=st.columns(2)
    with col1:
        index=st.button("Index",use_container_width=True)
    with col2:
        load=st.button("Load",use_container_width=True, help="Load existing index")

    if index and repo_url:
        with st.spinner("Cloning & indexing… this may take a few minutes."):
            try:
                vs=ingest_repo(repo_url,force_reindex=force_reindex)
                retriever=build_hybrid_retriever(vs)
                st.session_state.qa_instance=CodebaseQA(retriever)
                st.session_state.messages=[]
                st.session_state.indexed_repo=repo_url
                st.success(f"✅ Indexed: `{repo_url}`")
            except Exception as e:
                st.error(f"❌ Indexing failed: {e}")

    if load and repo_url:
        with st.spinner("Loading existing index…"):
            try:
                vs=load_vectorstore(repo_url)
                retriever=build_hybrid_retriever(vs)
                st.session_state.qa_instance=CodebaseQA(retriever)
                st.session_state.messages=[]
                st.session_state.indexed_repo=repo_url
                st.success(f"✅ Indexed: `{repo_url}`")
            except Exception as e:
                st.error(f"No index found. Click 🚀 Index to create one.")

    st.divider()

    if "indexed_repo" in st.session_state:
        st.caption(f"**Active repo:**")
        st.code(st.session_state.indexed_repo, language=None)

        if st.button("Clear Chat",use_container_width=True):
            st.session_state.messages=[]
            if _getqa():
                _getqa().reset_memory()
            st.rerun()

    st.divider()

    st.subheader("💡 Example Questions")
    examples = [
        "Where is the authentication logic?",
        "How are database models defined?",
        "What is the main entry point?",
        "How are API routes registered?",
        "Where is error handling implemented?",
    ]
    for ex in examples:
        if st.button(ex,use_container_width=True,key=f"ex_{ex[:20]}"):
            st.session_state.pending_question=ex
    

st.header("💬 Chat with Your Codebase")
messages=_get_messages()
qa=_getqa()


if not qa:
    st.info("👈 Enter a GitHub URL and click **Index** (or **Load**) to get started.")
else:
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                source_count=msg["sources"]
                with st.expander(f"{len(source_count)} source(s)"):
                    for src in msg["sources"]:
                        st.markdown(
                            f'<div class="source-card">'
                            f'<span class="file">{src["file"]}</span> &nbsp;'
                            f'<span class="lines">lines {src["lines"]}</span> &nbsp;'
                            f'<span class="lang">[{src["language"]}]</span>'
                            f"</div>",
                            unsafe_allow_html=True,
                        )

    pending=st.session_state.pop("pending_question",None)
    user_input=pending or st.chat_input("Ask Your question about CodeBase")

    if user_input:
        messages.append({"role":"user","content":user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Searching CodeBase"):
                result:QAResult=qa.ask(user_input)

            st.markdown(result.answer)
            if result.sources:
                result_count=result.sources
                with st.expander(f"{len(result_count)} source(s)"):
                    for src in result.sources:
                        st.markdown(
                            f'<div class="source-card">'
                            f'<span class="file">{src["file"]}</span> &nbsp;'
                            f'<span class="lines">lines {src["lines"]}</span> &nbsp;'
                            f'<span class="lang">[{src["language"]}]</span>'
                            f"</div>",
                            unsafe_allow_html=True,
                        )
        
        messages.append(
            {"role":"assistant",
             "content":result.answer,
             "sources":result.sources,
            }
        )









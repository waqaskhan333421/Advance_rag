"""Streamlit Web UI for Digital Islamic Library AI with Multi-Provider (Gemini & NVIDIA NIM) Support."""

import hashlib
import os
from pathlib import Path
from typing import Optional

import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Digital Islamic Library AI | PDF Q&A",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern polished design
st.markdown(
    """
    <style>
    /* Global styling */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #10b981, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 1.2rem;
    }
    .query-box {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-left: 4px solid #10b981;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 12px;
    }
    .query-title {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #10b981;
        margin-bottom: 4px;
    }
    .query-content {
        font-size: 0.92rem;
        color: #e2e8f0;
    }
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 4px;
    }
    .badge-primary { background-color: #1e40af; color: #93c5fd; }
    .badge-success { background-color: #065f46; color: #6ee7b7; }
    .badge-warning { background-color: #854d0e; color: #fde047; }
    .badge-purple { background-color: #581c87; color: #d8b4fe; }
    .badge-emerald { background-color: #064e3b; color: #a7f3d0; }
    .badge-nvidia { background-color: #155e75; color: #67e8f9; }
    .badge-gemini { background-color: #4338ca; color: #c7d2fe; }
    .badge-groq   { background-color: #c2410c; color: #ffedd5; }
    
    .source-box {
        background-color: #0f172a;
        border-left: 4px solid #3b82f6;
        padding: 12px 14px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 12px;
        font-size: 0.88rem;
    }
    .pdf-viewer-header {
        background-color: #1e293b;
        padding: 8px 12px;
        border-radius: 6px;
        border: 1px solid #334155;
        margin-bottom: 10px;
        font-size: 0.9rem;
        font-weight: 600;
    }
    /* Technique selector */
    .technique-header {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748b;
        margin: 10px 0 4px 0;
    }
    /* Retrieved document card */
    .doc-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-left: 5px solid #10b981;
        border-radius: 8px;
        padding: 16px 18px;
        margin-bottom: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .doc-card-title {
        font-size: 0.92rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 6px;
    }
    .doc-card-meta {
        font-size: 0.82rem;
        color: #94a3b8;
        margin-bottom: 10px;
        line-height: 1.6;
    }
    .doc-card-text {
        font-size: 0.88rem;
        color: #e2e8f0;
        line-height: 1.65;
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 12px 14px;
        margin-top: 8px;
        white-space: pre-wrap;
        word-break: break-word;
    }
    .badge-dense  { background-color: #1e3a5f; color: #7dd3fc; }
    .badge-sparse { background-color: #1a3a2a; color: #6ee7b7; }
    .badge-rrf    { background-color: #3b1f5e; color: #d8b4fe; }
    .badge-rerank { background-color: #7c2d12; color: #fdba74; }
    .badge-visual { background-color: #0f3460; color: #93c5fd; }
    /* Centered Q&A layout */
    .qa-center-wrap {
        max-width: 820px;
        margin: 0 auto;
    }
    .qa-center-wrap .stChatMessage {
        max-width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

from src.config import CONFIG
from src.pipeline import AdvancedRAGPipeline, PipelineResult

NO_ANSWER_PHRASE = "I don't know based on the provided documents"
CHECK_ALL_LABEL = "🌐 Check All (Search Entire Library)"


def doc_id_from_pdf(pdf_path: Path) -> Optional[str]:
    """Compute doc_id (MD5 prefix) for a PDF file."""
    try:
        return hashlib.md5(pdf_path.read_bytes()).hexdigest()[:16]
    except Exception:
        return None


def is_answer_found(result: PipelineResult) -> bool:
    """Return True when retrieval produced usable context and a non-fallback answer."""
    if not result.final_chunks:
        return False
    return NO_ANSWER_PHRASE.lower() not in result.answer.lower()


def run_scoped_query(
    pipeline: AdvancedRAGPipeline,
    question: str,
    selected_doc_id: Optional[str],
    **query_kwargs,
) -> PipelineResult:
    """Query a single resource or search across the entire library in a single pass."""
    if selected_doc_id:
        return pipeline.query(
            question,
            metadata_filter={"doc_id": {"$eq": selected_doc_id}},
            **query_kwargs,
        )

    return pipeline.query(question, metadata_filter=None, **query_kwargs)


@st.cache_resource(show_spinner="Initializing Digital Islamic Library AI (Models, Pinecone & BM25)...")
def get_pipeline() -> AdvancedRAGPipeline:
    """Initialize and cache the AdvancedRAGPipeline instance."""
    return AdvancedRAGPipeline()


def find_pdf_path(doc_id: Optional[str] = None, file_name: Optional[str] = None) -> Optional[Path]:
    """Find absolute path to a PDF by doc_id or file_name with smart fallbacks."""
    pdf_dir = Path(CONFIG.paths.documents_dir)
    if not pdf_dir.exists():
        return None

    # 1. Exact or stem match by file_name
    if file_name:
        direct = pdf_dir / file_name
        if direct.exists():
            return direct
        target_stem = Path(file_name).stem.lower().strip()
        for p in pdf_dir.glob("*.pdf"):
            if p.name.lower() == file_name.lower() or p.stem.lower().strip() == target_stem:
                return p

    # 2. Check by doc_id (MD5 hash)
    if doc_id:
        for p in pdf_dir.glob("*.pdf"):
            try:
                h = hashlib.md5(p.read_bytes()).hexdigest()[:16]
                if h == doc_id:
                    return p
            except Exception:
                pass

    # 3. Fallback to first available PDF
    all_pdfs = list(pdf_dir.glob("*.pdf"))
    if all_pdfs:
        return all_pdfs[0]

    return None


def main():
    # Initialize pipeline
    try:
        pipeline = get_pipeline()
    except Exception as e:
        st.error(f"⚠️ Error initializing RAG pipeline: {str(e)}")
        st.info("Please check your `.env` file and API keys (`GEMINI_API_KEY`, `NVIDIA_API_KEY`, `PINECONE_API_KEY`).")
        st.stop()

    # Session State Setup
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "selected_doc_path" not in st.session_state:
        st.session_state.selected_doc_path = None
    # Default all techniques ON
    if "techniques" not in st.session_state:
        st.session_state.techniques = {
            "query_rewriting": True,
            "hyde": True,
            "dense": True,
            "sparse": True,
            "rrf": True,
            "reranking": True,
            "compression": True,
        }

    # Sidebar: Model Selection, System Status, Document Management, Filters, and Config
    with st.sidebar:
        st.markdown("### 🤖 Model & Provider Selection")
        
        # Provider selector
        provider_choice = st.radio(
            "Select LLM Provider:",
            options=["Groq (Ultra-Fast)", "Google Gemini", "NVIDIA NIM"],
            index=0,
            help="Switch between Groq, Google Gemini, and NVIDIA NIM models.",
        )
        if provider_choice == "Groq (Ultra-Fast)":
            selected_provider = "groq"
        elif provider_choice == "Google Gemini":
            selected_provider = "gemini"
        else:
            selected_provider = "nvidia"

        # Model Selector based on provider
        if selected_provider == "groq":
            groq_models = [
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "mixtral-8x7b-32768",
                "gemma2-9b-it",
                "deepseek-r1-distill-llama-70b",
                "Custom...",
            ]
            chosen_gr = st.selectbox(
                "Groq Model:",
                options=groq_models,
                index=0,
                help="Select ultra-fast LLM hosted on Groq LPU hardware.",
            )
            if chosen_gr == "Custom...":
                selected_model = st.text_input("Enter Groq Model Identifier:", value="llama-3.3-70b-versatile")
            else:
                selected_model = chosen_gr
        elif selected_provider == "gemini":
            gemini_models = [
                "gemini-2.5-flash",
                "gemini-3.5-flash-lite",
                "gemini-3.1-flash-lite",
                "gemini-3.5-flash",
                "gemini-3.6-flash",
                "gemini-3.7-flash",
            ]
            selected_model = st.selectbox(
                "Gemini Model:",
                options=gemini_models,
                index=0,
                help="Select Gemini model version for query transformation & response generation.",
            )
        else:
            nvidia_models = [
                "nvidia/llama-3.1-nemotron-70b-instruct",
                "mistralai/mistral-large-2-instruct",
                "mistralai/mistral-7b-instruct-v0.3",
                "nvidia/nemotron-4-340b-instruct",
                "meta/llama-3.2-11b-vision-instruct",
                "Custom...",
            ]
            chosen_nv = st.selectbox(
                "NVIDIA NIM Model:",
                options=nvidia_models,
                index=0,
                help="Select NVIDIA hosted LLM via https://integrate.api.nvidia.com/v1.",
            )
            if chosen_nv == "Custom...":
                selected_model = st.text_input("Enter NVIDIA Model Identifier:", value="nvidia/llama-3.1-nemotron-70b-instruct")
            else:
                selected_model = chosen_nv

        # Temperature & Tokens
        with st.expander("🎛️ Generation Hyperparameters", expanded=False):
            temp_val = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
            max_tokens_val = st.slider("Max Output Tokens", min_value=256, max_value=4096, value=2048, step=256)

        # API Keys Status
        st.markdown("---")
        groq_key = os.getenv("GROQ_API_KEY", "")
        gemini_key = os.getenv(CONFIG.models.gemini.api_key_env, os.getenv("GEMINI_API_KEY", ""))
        nvidia_key = os.getenv(CONFIG.models.nvidia.api_key_env, os.getenv("NVIDIA_API_KEY", ""))
        pinecone_key = os.getenv(CONFIG.pinecone.api_key_env, os.getenv("PINECONE_API_KEY", ""))
        
        with st.expander("🔑 API Key Status", expanded=False):
            if groq_key:
                st.success("✅ Groq API Key: Active")
            else:
                st.warning("⚠️ Groq API Key: Missing (.env)")

            if gemini_key:
                st.success("✅ Gemini API Key: Active")
            else:
                st.error("❌ Gemini API Key: Missing")

            if pinecone_key:
                st.success("✅ Pinecone API Key: Active")
            else:
                st.error("❌ Pinecone API Key: Missing")

            if nvidia_key:
                st.info("ℹ️ NVIDIA API Key: Optional")

        # ── Technique Selector ────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🛠️ RAG Technique Selector")
        st.caption("Enable or disable individual pipeline stages per query.")

        tech = st.session_state.techniques

        st.markdown('<div class="technique-header">🔤 Query Enhancement</div>', unsafe_allow_html=True)
        tech["query_rewriting"] = st.checkbox(
            "Query Rewriting",
            value=tech["query_rewriting"],
            help="Rewrites the user query using the LLM to improve retrieval precision.",
        )
        tech["hyde"] = st.checkbox(
            "HyDE — Hypothetical Document Embeddings",
            value=tech["hyde"],
            help="Generates a synthetic 'ideal answer' document and embeds it to improve dense retrieval.",
        )

        st.markdown('<div class="technique-header">🔍 Retrieval</div>', unsafe_allow_html=True)
        tech["dense"] = st.checkbox(
            "Dense Retrieval (Pinecone Vector Search)",
            value=tech["dense"],
            help="Semantic similarity search using Gemini embeddings stored in Pinecone.",
        )
        tech["sparse"] = st.checkbox(
            "Sparse Retrieval (BM25 Lexical)",
            value=tech["sparse"],
            help="Keyword-based lexical matching using BM25 over ingested documents.",
        )
        tech["rrf"] = st.checkbox(
            "Hybrid Fusion — RRF",
            value=tech["rrf"],
            help="Reciprocal Rank Fusion merges dense + sparse rankings into a unified ranked list.",
            disabled=not (tech["dense"] and tech["sparse"]),
        )

        st.markdown('<div class="technique-header">⚙️ Post-Retrieval</div>', unsafe_allow_html=True)
        tech["reranking"] = st.checkbox(
            "Cross-Encoder Reranking (MS-MARCO)",
            value=tech["reranking"],
            help="Uses a cross-encoder model to precisely re-score retrieved chunks for top-K selection.",
        )
        tech["compression"] = st.checkbox(
            "Context Compression (LLM Filter)",
            value=tech["compression"],
            help="LLM removes irrelevant sentences from chunks before passing to generation.",
        )

        # Active technique badges
        active = [k for k, v in tech.items() if v]
        badge_map = {
            "query_rewriting": ("badge-primary",   "Query Rewrite"),
            "hyde":            ("badge-purple",    "HyDE"),
            "dense":           ("badge-dense",     "Dense"),
            "sparse":          ("badge-sparse",    "BM25"),
            "rrf":             ("badge-rrf",       "RRF"),
            "reranking":       ("badge-rerank",    "Reranker"),
            "compression":     ("badge-success",   "Compression"),
        }
        badges_html = " ".join(
            f'<span class="badge {badge_map[k][0]}">{badge_map[k][1]}</span>'
            for k in active if k in badge_map
        )
        st.markdown(f'<div style="margin-top:8px">{badges_html}</div>', unsafe_allow_html=True)

        # Pipeline Configuration summary (collapsed)
        st.markdown("---")
        with st.expander("⚙️ System Configuration", expanded=False):
            st.write(f"**Provider:** `{selected_provider.upper()}`")
            st.write(f"**Model:** `{selected_model}`")
            st.write(f"**Vector DB:** Pinecone (`{CONFIG.pinecone.index_name}`)")
            st.write(f"**Cross-Encoder:** `{CONFIG.models.cross_encoder.model_name}`")
            st.write(f"**Dense Top-K:** `{CONFIG.retrieval.dense_top_k}`")
            st.write(f"**Sparse Top-K:** `{CONFIG.retrieval.sparse_top_k}`")
            st.write(f"**Rerank Top-K:** `{CONFIG.retrieval.rerank_top_k}`")

        # Chat actions
        st.markdown("---")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_result = None
            st.session_state.selected_doc_path = None
            st.session_state.source_resource_select = CHECK_ALL_LABEL
            st.rerun()

    # Main Area Tabs
    tab_chat, tab_docs, tab_arch = st.tabs(["💬 Q&A Chat", "📁 Document Management", "📊 Architecture & Stats"])

    # -------------------------------------------------------------
    # TAB 1: Q&A Chat
    # -------------------------------------------------------------
    with tab_chat:
        _, center_col, _ = st.columns([1, 4, 1])
        with center_col:
            st.markdown('<div class="main-header">Digital Islamic Library AI</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="sub-header">Multi-Provider PDF Q&A powered by Google Gemini & NVIDIA NIM, Pinecone Hybrid Search, RRF Fusion, and Cross-Encoder Reranking.</div>',
                unsafe_allow_html=True,
            )

            # Source resource: Check All (default) or a single PDF
            pdf_dir = Path(CONFIG.paths.documents_dir)
            all_pdfs = list(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []

            resource_options: dict[str, Optional[str]] = {CHECK_ALL_LABEL: None}
            pdf_path_by_label: dict[str, Path] = {}
            for pdf_path in all_pdfs:
                label = pdf_path.name
                resource_options[label] = doc_id_from_pdf(pdf_path)
                pdf_path_by_label[label] = pdf_path

            resource_labels = list(resource_options.keys())

            # Selection persists across reruns via the widget key ("source_resource_select"),
            # so no separate mirror / index bookkeeping is needed. Defaults to the first
            # option (Check All). The Clear Chat button resets this key to CHECK_ALL_LABEL.
            selected_resource_label = st.selectbox(
                "Source Resource:",
                options=resource_labels if resource_options else ["No PDFs available"],
                help="Choose Check All to search every indexed document, or pick one PDF to search only that file.",
                key="source_resource_select",
                disabled=not resource_options,
            )

            selected_doc_id = resource_options.get(selected_resource_label)

            if selected_doc_id and selected_resource_label in pdf_path_by_label:
                st.session_state.selected_doc_path = str(pdf_path_by_label[selected_resource_label])
                if st.session_state.get("last_selected_resource") != selected_doc_id:
                    st.session_state.last_selected_resource = selected_doc_id
            elif not st.session_state.selected_doc_path and all_pdfs:
                st.session_state.selected_doc_path = str(all_pdfs[0])

            if selected_provider == "groq":
                provider_badge = f'<span class="badge badge-groq">⚡ Provider: Groq ({selected_model})</span>'
            elif selected_provider == "gemini":
                provider_badge = f'<span class="badge badge-gemini">🤖 Provider: Gemini ({selected_model})</span>'
            else:
                provider_badge = f'<span class="badge badge-nvidia">🟢 Provider: NVIDIA NIM ({selected_model})</span>'

            if selected_doc_id:
                scope_badge = f'<span class="badge badge-warning">🎯 Resource: {selected_resource_label}</span>'
                st.markdown(f'<div style="margin-bottom: 10px;">{provider_badge} {scope_badge}</div>', unsafe_allow_html=True)
                st.info(f"🎯 **Targeted Search Active:** Query will search ONLY within **{selected_resource_label}**.")
            else:
                scope_badge = '<span class="badge badge-emerald">🌐 Scope: Check All — Entire Library</span>'
                st.markdown(f'<div style="margin-bottom: 10px;">{provider_badge} {scope_badge}</div>', unsafe_allow_html=True)

            for idx, msg in enumerate(st.session_state.messages):
                with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "📖"):
                    if msg["role"] == "assistant" and "result" in msg:
                        res = msg["result"]

                        if getattr(res, "failed_over", False):
                            req_prov = (res.requested_provider or "").upper()
                            act_prov = (res.provider_used or "").upper()
                            reason = res.primary_error or "the selected provider was unavailable"
                            st.warning(
                                f"⚠️ **Provider failover:** your selected **{req_prov}** "
                                f"(`{res.requested_model}`) could not answer, so this reply was "
                                f"generated with **{act_prov}** (`{res.model_used}`) instead.\n\n"
                                f"_Reason: {str(reason)[:200]}_"
                            )

                        st.markdown(
                            f"""
                            <div class="query-box">
                                <div class="query-title">🔄 Query Transformation ({res.provider_used.upper()}: {res.model_used})</div>
                                <div class="query-content">
                                    <strong>Original Query:</strong> <em>"{res.query}"</em><br/>
                                    <strong>Rewritten Search Query:</strong> <span style="color: #6ee7b7; font-weight: 600;">"{res.rewritten_query}"</span>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        st.markdown(msg["content"])

                        if hasattr(res, "latency_ms") and res.latency_ms:
                            lats = res.latency_ms
                            p_badge = "badge-groq" if res.provider_used == "groq" else ("badge-gemini" if res.provider_used == "gemini" else "badge-nvidia")
                            st.markdown(
                                f"""
                                <div style="margin-top: 8px; margin-bottom: 8px;">
                                    <span class="badge {p_badge}">⚙️ {res.provider_used.upper()}</span>
                                    <span class="badge badge-emerald">⏱️ Total: {lats.get('total', 0)}ms</span>
                                    <span class="badge badge-purple">🔄 Rewrite: {lats.get('query_transform', 0)}ms</span>
                                    <span class="badge badge-success">🔎 Retrieval: {lats.get('retrieval', 0)}ms</span>
                                    <span class="badge badge-warning">⚖️ Rerank: {lats.get('rerank', 0)}ms</span>
                                    <span class="badge badge-primary">✍️ Generation: {lats.get('generation', 0)}ms</span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        # ── Extracted Source Data Section ────────────────────────
                        with st.expander(
                            f"📖 Extracted Source Data & Citations ({len(res.final_chunks)} evidence passages used)",
                            expanded=True,
                        ):
                            if not res.final_chunks:
                                st.info("No source passages retrieved for this query.")
                            else:
                                st.markdown(
                                    f"""
                                    <div style="margin-bottom:12px; font-size:0.82rem; color:#94a3b8">
                                        🔵 Dense: <b>{res.dense_results}</b> &nbsp;|&nbsp;
                                        🟢 BM25: <b>{res.sparse_results}</b> &nbsp;|&nbsp;
                                        🟣 After RRF: <b>{res.fused_results}</b> &nbsp;|&nbsp;
                                        🟠 After Rerank: <b>{res.reranked_results}</b> &nbsp;|&nbsp;
                                        ✅ <b>{len(res.final_chunks)} Citations Extracted</b>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                                for c_idx, chunk in enumerate(res.final_chunks, 1):
                                    page        = chunk.get("page", 1)
                                    section     = chunk.get("section") or "General Content"
                                    score       = chunk.get("score", 0.0)
                                    text        = chunk.get("text", "")
                                    file_name   = chunk.get("file_name", "Document")
                                    chunk_id    = chunk.get("chunk_id", chunk.get("id", "—"))
                                    visual_path = chunk.get("visual_asset_path")

                                    method_badges = ""
                                    if tech.get("dense"):     method_badges += '<span class="badge badge-dense">Dense</span> '
                                    if tech.get("sparse"):    method_badges += '<span class="badge badge-sparse">BM25</span> '
                                    if tech.get("rrf"):       method_badges += '<span class="badge badge-rrf">RRF</span> '
                                    if tech.get("reranking"): method_badges += '<span class="badge badge-rerank">Reranked</span> '

                                    st.markdown(
                                        f"""
                                        <div class="doc-card">
                                            <div class="doc-card-title">
                                                <span>Source #{c_idx}</span>
                                                <span class="badge badge-primary">📗 {file_name}</span>
                                                <span class="badge badge-purple">📄 Page {page}</span>
                                                <span class="badge badge-success">⭐ Match Score: {score:.4f}</span>
                                            </div>
                                            <div class="doc-card-meta">
                                                <b>📑 Section:</b> {section} &nbsp;&nbsp;|&nbsp;&nbsp;
                                                <b>🗂 Chunk ID:</b> <code>{str(chunk_id)[:20]}</code><br/>
                                                <b>🔧 Retrieved via:</b> {method_badges}
                                            </div>
                                            <div style="font-size:0.78rem; font-weight:700; color:#10b981; text-transform:uppercase; margin-top:8px;">
                                                📝 Extracted Passage Text:
                                            </div>
                                            <div class="doc-card-text">{text}</div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )
                                    if visual_path:
                                        vp = Path(visual_path)
                                        if not vp.is_absolute():
                                            vp = Path(".") / vp
                                        if vp.exists():
                                            st.image(str(vp), caption=f"📊 Associated Visual Asset — {vp.name}", use_container_width=True)

                        with st.expander("🔍 HyDE (Hypothetical Document Synthesis)", expanded=False):
                            st.markdown("**Synthesized Technical Context:**")
                            st.info(res.hyde_doc)
                    else:
                        st.markdown(msg["content"])

            user_input = st.chat_input("Ask any question from the Islamic digital library...")
            if user_input:
                st.session_state.messages.append({"role": "user", "content": user_input})
                with st.spinner(f"Running pipeline via {selected_provider.upper()} ({selected_model})..."):
                    try:
                        result = run_scoped_query(
                            pipeline,
                            user_input,
                            selected_doc_id,
                            provider=selected_provider,
                            llm_model=selected_model,
                            temperature=temp_val,
                            max_tokens=max_tokens_val,
                            use_query_rewriting=tech["query_rewriting"],
                            use_hyde=tech["hyde"],
                            use_dense=tech["dense"],
                            use_sparse=tech["sparse"],
                            use_rrf=tech["rrf"],
                            use_reranking=tech["reranking"],
                            use_compression=tech["compression"],
                        )
                        st.session_state.last_result = result
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result.answer,
                            "result": result,
                        })
                        if getattr(result, "failed_over", False):
                            st.toast(
                                f"⚠️ {(result.requested_provider or '').upper()} unavailable — "
                                f"answered via {(result.provider_used or '').upper()} "
                                f"({result.model_used}) instead.",
                                icon="⚠️",
                            )
                    except Exception as e:
                        st.error(f"Error during query execution: {str(e)}")
                st.rerun()

    # -------------------------------------------------------------
    # TAB 2: Document Management & Ingestion
    # -------------------------------------------------------------
    with tab_docs:
        st.markdown("### 📁 Document Ingestion & Management")
        st.markdown("Upload PDFs to automatically extract layout, OCR scanned text, apply semantic chunking, and index into Pinecone & BM25.")

        doc_dir = Path(CONFIG.paths.documents_dir)
        doc_dir.mkdir(parents=True, exist_ok=True)

        col_upload, col_actions = st.columns([2, 1])

        with col_upload:
            uploaded_files = st.file_uploader(
                "Upload PDF Document(s)",
                type=["pdf"],
                accept_multiple_files=True,
                help="Upload one or more PDF files. They will be saved to your configured documents directory and indexed.",
            )

            if uploaded_files:
                if st.button("📥 Ingest Uploaded Files", type="primary"):
                    with st.status("Processing and indexing documents...", expanded=True) as status:
                        for uf in uploaded_files:
                            file_path = doc_dir / uf.name
                            with open(file_path, "wb") as f:
                                f.write(uf.getbuffer())
                            st.write(f"Saved: `{uf.name}`")

                        st.write("Running ingestion and embedding pipeline...")
                        pipeline.index_documents(pdf_dir=str(doc_dir), force_reindex=False)
                        status.update(label="✅ Ingestion and Indexing Complete!", state="complete", expanded=False)
                    st.success(f"Successfully processed {len(uploaded_files)} file(s)!")
                    st.rerun()

        with col_actions:
            st.markdown("#### Indexing Actions")
            st.write("Trigger index synchronization for all files in `./data/pdfs/`:")
            
            run_opt_b_sync = st.checkbox("Include Option B (Gemini Vision Image Extraction)", value=False, help="Runs Gemini Vision to extract & caption embedded PDF images before indexing.")

            if st.button("🔄 Sync New Documents", use_container_width=True):
                with st.spinner("Syncing documents..."):
                    pipeline.index_documents(pdf_dir=str(doc_dir), force_reindex=False, extract_images=run_opt_b_sync)
                st.success("Sync complete!")
                st.rerun()

            if st.button("⚠️ Force Full Re-Index", use_container_width=True, help="Re-chunks, re-embeds, and re-indexes all PDFs from scratch"):
                with st.spinner("Force re-indexing all documents..."):
                    pipeline.index_documents(pdf_dir=str(doc_dir), force_reindex=True, extract_images=run_opt_b_sync)
                st.success("Full re-indexing complete!")
                st.rerun()

        # Option B Dedicated Showcase Card
        st.markdown("---")
        st.markdown("#### 👁️ Option B: Multimodal Gemini Vision Processing")
        st.info(
            "**Option B Workflow:** Extracts embedded images/calligraphy/diagrams from all PDF books → "
            "captions them using **Gemini Vision** (OCR, Arabic translation, visual analysis) → "
            "writes a searchable PDF corpus (`extracted_images_corpus.pdf`) → embeds into Pinecone & BM25."
        )

        col_optb1, col_optb2 = st.columns([2, 1])
        with col_optb1:
            st.write("Run Gemini Vision image extraction pipeline separately or trigger immediate corpus generation.")
        with col_optb2:
            if st.button("🖼️ Run Option B Image Extraction", use_container_width=True, type="primary"):
                with st.status("Running Option B — Gemini Vision Image Extraction...", expanded=True) as status:
                    from src.image_extractor import run as run_image_extractor
                    st.write("Scanning PDFs for images...")
                    run_image_extractor(pdf_dir=str(doc_dir))
                    st.write("Re-indexing complete corpus including extracted image descriptions...")
                    pipeline.index_documents(pdf_dir=str(doc_dir), force_reindex=False)
                    status.update(label="✅ Option B Complete! Image Corpus PDF generated & indexed.", state="complete")
                st.success("Option B complete!")
                st.rerun()

        st.markdown("---")
        st.markdown("#### 📚 Ingested Documents")
        
        current_docs = pipeline.list_documents()
        if not current_docs:
            st.warning("No documents have been indexed yet. Upload a PDF above or run indexing.")
        else:
            for d in current_docs:
                with st.container():
                    c1, c2, c3, c4 = st.columns([3, 1, 1, 2])
                    c1.markdown(f"📄 **{d.get('file_name', 'Unknown')}**")
                    c2.markdown(f"🧩 **{d.get('chunks', 0)}** chunks")
                    c3.markdown(f"📖 **{d.get('max_page', 1)}** pages")
                    c4.markdown(f"`ID: {d.get('doc_id', 'N/A')}`")
                    st.divider()

    # -------------------------------------------------------------
    # TAB 3: Architecture & System Diagnostics
    # -------------------------------------------------------------
    with tab_arch:
        st.markdown("### 📊 Architecture & Pipeline Diagnostics")
        
        st.markdown(
            """
            ```
            ┌────────────────┐
            │   User Query   │
            └───────┬────────┘
                    │
            ┌───────▼─────────────────────────────────────────────────┐
            │ Query Transformation (Rewrite + HyDE via Gemini/NVIDIA) │
            └───────┬─────────────────────────────────────────────────┘
                    │
            ┌───────▼─────────────────────────────────────────────────┐
            │ Hybrid Retrieval                                        │
            │  ├─ Dense: Pinecone (Gemini Embeddings, top_k=30)       │
            │  └─ Sparse: BM25 (Lexical matching, top_k=30)           │
            └───────┬─────────────────────────────────────────────────┘
                    │
            ┌───────▼─────────────────────────────────────────────────┐
            │ Reciprocal Rank Fusion (RRF k=60) → Top-50 Candidates   │
            └───────┬─────────────────────────────────────────────────┘
                    │
            ┌───────▼─────────────────────────────────────────────────┐
            │ Cross-Encoder Reranker (MS-MARCO MiniLM) → Top-5 Chunks │
            └───────┬─────────────────────────────────────────────────┘
                    │
            ┌───────▼─────────────────────────────────────────────────┐
            │ Context Compression (LLM Sentence Relevance Filter)     │
            └───────┬─────────────────────────────────────────────────┘
                    │
            ┌───────▼─────────────────────────────────────────────────┐
            │ LLM Generation (Gemini 3.6 / NVIDIA NIM LLaMA 3.1 70B)  │
            └─────────────────────────────────────────────────────────┘
            ```
            """
        )

        st.markdown("#### 📈 Active Pipeline Parameters")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Dense Top-K", CONFIG.retrieval.dense_top_k)
            st.metric("Sparse Top-K", CONFIG.retrieval.sparse_top_k)
        with c2:
            st.metric("RRF Constant (k)", CONFIG.retrieval.rrf_k)
            st.metric("RRF Fusion Top-K", CONFIG.retrieval.fusion_top_k)
        with c3:
            st.metric("Reranker Final Top-K", CONFIG.retrieval.rerank_top_k)
            st.metric("Chunk Size / Overlap", f"{CONFIG.chunking.chunk_size} / {CONFIG.chunking.chunk_overlap}")


if __name__ == "__main__":
    main()

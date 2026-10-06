"""
app.py — Interactive Multi-Modal Video Retrieval Dashboard
==========================================================
Run with:
    streamlit run src/app.py

Two-stage retrieval pipeline
-----------------------------
Stage 1 — Qdrant cosine search:    text embedding → top-K candidate segments
Stage 2 — CrossAttentionFusion:    PyTorch cross-attention re-ranking on candidates
Final    — st.video playback at the highest-scoring timestamp.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# ── Ensure src/ sibling modules are importable ────────────────────────────────
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import streamlit as st
from loguru import logger
from transformers import CLIPModel, CLIPProcessor
from google import genai

from database import VectorStore
from model import CrossAttentionFusion
from pipeline import (
    RAW_VIDEO_DIR,
    TENSOR_CACHE_DIR,
    FUSED_DIM,
    extract_video_embeddings,
    list_indexed_videos,
    extract_frames_for_segment,
)

# ── Constants ─────────────────────────────────────────────────────────────────
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
TOP_K_QDRANT = 20          # Stage-1 candidates from cosine search
TOP_K_RERANK = 5           # Stage-2 cross-attention top results to display
ATTN_D_MODEL = 1024
ATTN_NUM_HEADS = 4

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MultiModal Search Engine",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #0d0d1a 0%, #111827 50%, #0f172a 100%);
        color: #e2e8f0;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid #2d3748;
    }

    /* Main title gradient */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #7c3aed, #06b6d4, #10b981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.2rem;
    }

    .hero-subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }

    /* Glassmorphism cards */
    .result-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
        transition: border-color 0.2s ease;
    }
    .result-card:hover {
        border-color: rgba(124, 58, 237, 0.4);
    }

    /* Score badge */
    .score-badge {
        display: inline-block;
        background: linear-gradient(90deg, #7c3aed, #06b6d4);
        color: white;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-left: 8px;
    }

    /* Stage labels */
    .stage-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #7c3aed;
        margin-bottom: 0.5rem;
    }

    /* Metric card */
    .metric-box {
        background: rgba(124, 58, 237, 0.1);
        border: 1px solid rgba(124, 58, 237, 0.25);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #7c3aed;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 0.2rem;
    }

    /* Search box override */
    div[data-testid="stTextInput"] input {
        background: rgba(255, 255, 255, 0.06) !important;
        border: 1px solid rgba(124, 58, 237, 0.4) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        font-size: 1rem !important;
        padding: 0.8rem 1rem !important;
    }

    /* Button override */
    div[data-testid="stButton"] > button {
        background: linear-gradient(90deg, #7c3aed, #06b6d4) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.6rem 2rem !important;
        transition: opacity 0.2s ease !important;
    }
    div[data-testid="stButton"] > button:hover {
        opacity: 0.88 !important;
    }

    .divider {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.07);
        margin: 1.5rem 0;
    }

    .timestamp-chip {
        display: inline-block;
        background: rgba(6, 182, 212, 0.15);
        border: 1px solid rgba(6, 182, 212, 0.3);
        color: #06b6d4;
        border-radius: 8px;
        padding: 2px 10px;
        font-size: 0.82rem;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Cached resource initialisation
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading CLIP text encoder…")
def load_clip_text_model() -> Tuple[CLIPProcessor, CLIPModel]:
    """Load and freeze the full CLIP model for text query encoding."""
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info("CLIP text encoder ready.")
    return processor, model


@st.cache_resource(show_spinner="Initialising cross-attention model…")
def load_fusion_model() -> CrossAttentionFusion:
    """Instantiate the CrossAttentionFusion module (CPU-ready, eval mode)."""
    model = CrossAttentionFusion(
        input_dim=FUSED_DIM,
        d_model=ATTN_D_MODEL,
        num_heads=ATTN_NUM_HEADS,
        dropout=0.0,
    )
    
    # Initialize with identity matrices for zero-shot evaluation
    # This prevents random Xavier weights from destroying the retrieval ranking
    model.W_q.weight.data = torch.eye(FUSED_DIM)
    model.W_k.weight.data = torch.eye(FUSED_DIM)
    model.W_v.weight.data = torch.eye(FUSED_DIM)
    model.W_o.weight.data = torch.eye(FUSED_DIM)
    
    model.eval()
    logger.info("CrossAttentionFusion model ready.")
    return model


@st.cache_resource(show_spinner="Connecting to vector database…")
def load_vector_store() -> VectorStore:
    """Open (or create) the local Qdrant collection."""
    return VectorStore()


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_text_query(query: str) -> List[float]:
    """
    Encode a natural-language string through CLIP's text branch and project
    to FUSED_DIM (1024) by repeating the 512-dim text embedding.

    The projection mirrors the visual + audio concatenation in pipeline.py:
      fused = [normalized_visual (512) || normalized_audio (512)]
    For query matching we use:
      query_fused = [normalized_text (512) || normalized_text (512)]
    This ensures cosine similarity operates in the same embedding space.
    """
    processor, model = load_clip_text_model()
    device = next(model.parameters()).device
    inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
    text_features = model.get_text_features(**inputs)  # (1, 512)
    text_features = F.normalize(text_features, dim=-1)  # unit-normalise
    # Expand to (1024,) so vector dimensions match stored segments
    # Whisper audio features are not aligned with CLIP text features, so pad audio half with zeros
    # to avoid random noise during cosine similarity search.
    query_1024 = torch.cat([text_features, torch.zeros_like(text_features)], dim=-1).squeeze(0)  # (1024,)
    return query_1024.cpu().float().tolist()


def fetch_candidate_tensors(
    hits: List[Dict[str, Any]],
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """
    Reload the .pt tensor cache for each candidate hit and reconstruct
    the segment embedding matrix for cross-attention re-ranking.

    Parameters
    ----------
    hits : list of Qdrant search result dicts

    Returns
    -------
    segment_matrix : torch.Tensor  shape (N_candidates, FUSED_DIM)
    valid_hits     : filtered list matching the matrix rows
    """
    tensors: List[torch.Tensor] = []
    valid_hits: List[Dict[str, Any]] = []

    for hit in hits:
        video_name = hit["video_name"]
        seg_idx = hit["segment_index"]
        cache_path = TENSOR_CACHE_DIR / f"{video_name}.pt"

        if not cache_path.exists():
            logger.warning(f"Cache missing for '{video_name}' — skipping hit.")
            continue

        saved = torch.load(cache_path, map_location="cpu", weights_only=False)
        if seg_idx < 0 or seg_idx >= len(saved):
            logger.warning(
                f"Segment index {seg_idx} out of range for '{video_name}' "
                f"(cache has {len(saved)} entries) — skipping."
            )
            continue

        embed = saved[seg_idx]["embedding"]  # (1024,)
        if embed.shape[0] != FUSED_DIM:
            logger.warning(
                f"Unexpected embedding dim {embed.shape[0]} for "
                f"'{video_name}' seg {seg_idx} — skipping."
            )
            continue

        tensors.append(embed)
        valid_hits.append(hit)

    if not tensors:
        return torch.empty(0, FUSED_DIM), []

    return torch.stack(tensors, dim=0), valid_hits  # (N, 1024)


def format_timestamp(seconds: float) -> str:
    """Convert float seconds to MM:SS.mm display string."""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:05.2f}"


def find_video_file(video_name: str) -> Optional[Path]:
    """
    Locate the .mp4 file for a given video stem in raw_videos/.
    Returns None if the file does not exist on disk.
    """
    for ext in [".mp4", ".MP4", ".mov", ".MOV", ".avi", ".AVI"]:
        candidate = RAW_VIDEO_DIR / f"{video_name}{ext}"
        if candidate.exists():
            return candidate
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — Video Index Management
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar(store: VectorStore) -> None:
    """Render the sidebar with indexing controls and collection statistics."""
    st.sidebar.markdown("## 🗂️ Video Index")
    st.sidebar.markdown("---")

    # ── Gemini Configuration ──────────────────────────────────────────────────
    st.sidebar.markdown("### 🤖 Video QA Agent (Gemini)")
    gemini_key = st.sidebar.text_input("Gemini API Key", type="password", help="Get a free key from Google AI Studio")
    enable_rag = st.sidebar.checkbox("Enable Video QA", value=False, disabled=not gemini_key, help="Ask questions about the video content.")
    if gemini_key:
        st.session_state["gemini_client"] = genai.Client(api_key=gemini_key)
    st.session_state["enable_rag"] = enable_rag

    st.sidebar.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── Collection stats ──────────────────────────────────────────────────────
    try:
        info = store.collection_info()
        st.sidebar.markdown(
            f"""
            <div class='metric-box'>
                <div class='metric-value'>{info['total_points'] or 0}</div>
                <div class='metric-label'>Indexed Segments</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception:
        st.sidebar.warning("Could not fetch collection info.")

    st.sidebar.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── File uploader ─────────────────────────────────────────────────────────
    st.sidebar.markdown("### ➕ Index New Video")
    uploaded = st.sidebar.file_uploader(
        "Upload MP4 file",
        type=["mp4", "mov", "avi"],
        help="Upload a video to extract and index its multi-modal embeddings.",
        key="video_uploader",
    )

    if uploaded is not None:
        save_path = RAW_VIDEO_DIR / uploaded.name
        if not save_path.exists():
            with open(save_path, "wb") as f:
                f.write(uploaded.getbuffer())
            st.sidebar.success(f"Saved: {uploaded.name}")

        if st.sidebar.button("🚀 Extract & Index", key="index_btn"):
            with st.spinner(f"Processing '{uploaded.name}'…"):
                try:
                    video_embeddings = extract_video_embeddings(str(save_path))
                    n_upserted = store.upsert_segments(video_embeddings)
                    st.sidebar.success(
                        f"✅ Indexed {n_upserted} segments from '{uploaded.name}'"
                    )
                    st.rerun()
                except Exception as exc:
                    st.sidebar.error(f"Indexing failed: {exc}")

    st.sidebar.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── Indexed video list ────────────────────────────────────────────────────
    st.sidebar.markdown("### 📋 Indexed Videos")
    indexed = list_indexed_videos()
    if indexed:
        for name in indexed:
            exists = find_video_file(name) is not None
            icon = "🎬" if exists else "⚠️"
            tooltip = "" if exists else " (source file missing)"
            st.sidebar.markdown(f"{icon} `{name}`{tooltip}")
    else:
        st.sidebar.info("No videos indexed yet.")


# ─────────────────────────────────────────────────────────────────────────────
# Main search interface
# ─────────────────────────────────────────────────────────────────────────────

def render_search_results(
    query: str,
    store: VectorStore,
    fusion_model: CrossAttentionFusion,
) -> None:
    """
    Execute the two-stage retrieval pipeline and render results.
    Stage 1: Qdrant cosine search → top-K candidate segments.
    Stage 2: CrossAttentionFusion re-ranking → top results.
    """
    st.markdown("<div class='stage-label'>⚡ Stage 1 — Encoding Query</div>", unsafe_allow_html=True)
    with st.spinner("Encoding query with CLIP text encoder…"):
        query_vector = encode_text_query(query)
    st.success(f"Query encoded → ({len(query_vector)},) vector ✓")

    # ── Stage 1: Qdrant cosine search ─────────────────────────────────────────
    st.markdown("<div class='stage-label'>🔍 Stage 1 — Cosine Vector Search</div>", unsafe_allow_html=True)
    with st.spinner("Searching vector database…"):
        hits = store.search(query_vector, top_k=TOP_K_QDRANT)

    if not hits:
        st.warning(
            "No results found. Make sure you have indexed at least one video first. "
            "Use the sidebar to upload and index MP4 files."
        )
        return

    st.markdown(
        f"Found **{len(hits)}** candidate segments (cosine similarity ranking).",
        unsafe_allow_html=True,
    )

    # ── Stage 2: Reload candidate tensors ─────────────────────────────────────
    st.markdown("<div class='stage-label'>🧠 Stage 2 — Cross-Attention Re-Ranking</div>", unsafe_allow_html=True)
    with st.spinner("Loading candidate tensors for re-ranking…"):
        segment_matrix, valid_hits = fetch_candidate_tensors(hits)

    if segment_matrix.shape[0] == 0:
        st.error(
            "Could not load embeddings for any candidate segments. "
            "Tensor cache files may be missing — re-index your videos."
        )
        return

    # Cross-attention re-ranking
    query_tensor = torch.tensor(query_vector, dtype=torch.float32)  # (1024,)
    with st.spinner("Running cross-attention fusion…"):
        top_indices, top_scores = fusion_model.rank_segments(
            query_tensor, segment_matrix, top_k=TOP_K_RERANK
        )

    top_indices_list = top_indices.tolist()
    top_scores_list = top_scores.tolist()

    st.success(
        f"Re-ranking complete. Showing top **{len(top_indices_list)}** results. ✓"
    )
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Stage 3: Video QA (Gemini RAG) ────────────────────────────────────────
    if st.session_state.get("enable_rag") and valid_hits:
        st.markdown("<div class='stage-label'>🤖 Stage 3 — Video Question Answering (Gemini)</div>", unsafe_allow_html=True)
        with st.spinner("Watching video segments and generating answer…"):
            try:
                # Collect frames from top 3 segments
                all_frames = []
                for rank, (idx, score) in enumerate(zip(top_indices_list[:3], top_scores_list[:3])):
                    hit = valid_hits[idx]
                    video_file = find_video_file(hit["video_name"])
                    if video_file:
                        frames = extract_frames_for_segment(
                            str(video_file), 
                            hit["timestamp_start"], 
                            hit["timestamp_end"] - hit["timestamp_start"],
                            num_frames=3
                        )
                        all_frames.extend(frames)
                
                if all_frames:
                    client = st.session_state.get("gemini_client")
                    if client:
                        prompt = f"You are a helpful video assistant. Based on these frames extracted from the most relevant video clips, please answer the user's question.\nQuestion: {query}"
                        
                        response = client.models.generate_content_stream(
                            model='gemini-2.5-flash',
                            contents=[prompt] + all_frames
                        )
                        
                        st.markdown("### 💡 Agent Answer")
                        response_container = st.empty()
                        full_response = ""
                        for chunk in response:
                            if chunk.text:
                                full_response += chunk.text
                                response_container.info(full_response + "▌")
                        response_container.info(full_response)
                    else:
                        st.error("Gemini API Client not initialized. Please re-enter your API key.")
                else:
                    st.warning("Could not extract frames for Video QA.")
            except Exception as e:
                st.error(f"Video QA failed: {str(e)}")
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Render results ────────────────────────────────────────────────────────
    st.markdown("## 🎯 Top Results")

    for rank, (idx, score) in enumerate(zip(top_indices_list, top_scores_list), start=1):
        hit = valid_hits[idx]
        video_name = hit["video_name"]
        t_start = hit["timestamp_start"]
        t_end = hit["timestamp_end"]
        cosine_score = hit["score"]

        video_file = find_video_file(video_name)

        with st.container():
            st.markdown(
                f"""
                <div class='result-card'>
                    <b>#{rank} — {video_name}</b>
                    <span class='score-badge'>Attn: {score:.4f}</span>
                    <br>
                    <span class='timestamp-chip'>⏱ {format_timestamp(t_start)} → {format_timestamp(t_end)}</span>
                    &nbsp;&nbsp;
                    <span style='color:#94a3b8; font-size:0.85rem;'>Cosine: {cosine_score:.4f}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if video_file is not None:
                with open(video_file, "rb") as vf:
                    video_bytes = vf.read()
                st.video(
                    video_bytes,
                    start_time=int(t_start),
                    format="video/mp4",
                )
            else:
                st.warning(
                    f"Source file for '{video_name}' not found in `data/raw_videos/`. "
                    "Re-upload the video to enable playback."
                )

            st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# Main application entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Hero header ───────────────────────────────────────────────────────────
    st.markdown(
        "<h1 class='hero-title'>🎬 Multi-Modal Video Retrieval</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p class='hero-subtitle'>"
        "Search your video library using natural language — "
        "powered by CLIP · Whisper · Cross-Attention Fusion · Qdrant"
        "</p>",
        unsafe_allow_html=True,
    )

    # ── Load shared resources ─────────────────────────────────────────────────
    store = load_vector_store()
    fusion_model = load_fusion_model()
    # Warm up CLIP text model
    load_clip_text_model()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    render_sidebar(store)

    # ── Search bar ────────────────────────────────────────────────────────────
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        query = st.text_input(
            label="Search Query",
            placeholder="e.g. 'a person running on the beach at sunset'",
            label_visibility="collapsed",
            key="search_input",
        )
    with col_btn:
        search_clicked = st.button("Search 🔍", key="search_btn", use_container_width=True)

    # ── Pipeline info expander ────────────────────────────────────────────────
    with st.expander("ℹ️ How it works", expanded=False):
        st.markdown(
            """
            **Two-Stage Retrieval Pipeline:**

            1. **CLIP Text Encoding** — Your query is encoded into a 1024-dimensional
               fused embedding space using `openai/clip-vit-base-patch32`.

            2. **Stage 1: Qdrant Cosine Search** — The query vector is compared against
               all stored segment embeddings using approximate nearest-neighbor search
               (HNSW index, Cosine distance) to retrieve the top candidates quickly.

            3. **Stage 2: Cross-Attention Fusion** — The candidate segment tensors are
               loaded into memory and fed through a custom `CrossAttentionFusion` PyTorch
               module. The text query acts as the attention Query (Q), while segment
               embeddings serve as both Keys (K) and Values (V). The formula used is:

               `Attention(Q, K, V) = softmax((Q @ Kᵀ) / √d_k) @ V`

            4. **Ranked Playback** — Results are ranked by attention weight and rendered
               with `st.video` starting at the exact matching timestamp.

            **Indexing Pipeline:**
            - Videos are chunked into 2-second segments.
            - 3 frames are sampled per segment → CLIP ViT-B/32 visual encoder → (512,).
            - Audio is extracted via ffmpeg → Whisper encoder → (512,).
            - Embeddings are concatenated → (1024,) and cached to disk.
            """
        )

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Execute search ────────────────────────────────────────────────────────
    if search_clicked or (query and st.session_state.get("_prev_query") != query):
        if not query.strip():
            st.warning("Please enter a search query.")
        else:
            st.session_state["_prev_query"] = query
            render_search_results(query.strip(), store, fusion_model)
    elif not list_indexed_videos():
        st.markdown(
            """
            <div style='text-align:center; padding: 4rem 2rem; color: #475569;'>
                <div style='font-size: 4rem;'>🎬</div>
                <h3 style='color: #64748b;'>No videos indexed yet</h3>
                <p>Upload MP4 files using the sidebar to get started.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style='text-align:center; padding: 4rem 2rem; color: #475569;'>
                <div style='font-size: 4rem;'>🔍</div>
                <h3 style='color: #64748b;'>Enter a query above to search</h3>
                <p>Use natural language to describe what you're looking for.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()

import os
import sys
from pathlib import Path

# ── Ensure src/ sibling modules are importable ────────────────────────────────
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
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
TOP_K_QDRANT = 20
TOP_K_RERANK = 5
ATTN_D_MODEL = 1024
ATTN_NUM_HEADS = 4

# ── App & Middleware ──────────────────────────────────────────────────────────
app = FastAPI(title="Multi-Modal Video API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount raw videos for streaming
app.mount("/api/videos", StaticFiles(directory=RAW_VIDEO_DIR), name="videos")

# ── Global ML State ───────────────────────────────────────────────────────────
ML_STATE = {
    "clip_processor": None,
    "clip_model": None,
    "store": None,
    "fusion_model": None,
}

@app.on_event("startup")
def load_models():
    """Load models into memory on startup."""
    logger.info("Loading CLIP text encoder...")
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ML_STATE["clip_processor"] = processor
    ML_STATE["clip_model"] = model

    logger.info("Loading CrossAttentionFusion...")
    fusion = CrossAttentionFusion(
        input_dim=FUSED_DIM,
        d_model=ATTN_D_MODEL,
        num_heads=ATTN_NUM_HEADS,
        dropout=0.0,
    )
    fusion.W_q.weight.data = torch.eye(FUSED_DIM)
    fusion.W_k.weight.data = torch.eye(FUSED_DIM)
    fusion.W_v.weight.data = torch.eye(FUSED_DIM)
    fusion.W_o.weight.data = torch.eye(FUSED_DIM)
    fusion.eval()
    ML_STATE["fusion_model"] = fusion

    logger.info("Connecting to Vector DB...")
    ML_STATE["store"] = VectorStore()
    logger.info("Backend ready.")

# ── Utility Functions ─────────────────────────────────────────────────────────

@torch.no_grad()
def encode_text_query(query: str) -> List[float]:
    processor = ML_STATE["clip_processor"]
    model = ML_STATE["clip_model"]
    device = next(model.parameters()).device
    inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
    text_features = model.get_text_features(**inputs)
    text_features = F.normalize(text_features, dim=-1)
    query_1024 = torch.cat([text_features, torch.zeros_like(text_features)], dim=-1).squeeze(0)
    return query_1024.cpu().float().tolist()

def fetch_candidate_tensors(hits: List[Dict[str, Any]]) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    tensors = []
    valid_hits = []
    for hit in hits:
        video_name = hit["video_name"]
        seg_idx = hit["segment_index"]
        cache_path = TENSOR_CACHE_DIR / f"{video_name}.pt"
        if not cache_path.exists():
            continue
        saved = torch.load(cache_path, map_location="cpu", weights_only=False)
        if seg_idx < 0 or seg_idx >= len(saved):
            continue
        embed = saved[seg_idx]["embedding"]
        if embed.shape[0] != FUSED_DIM:
            continue
        tensors.append(embed)
        valid_hits.append(hit)
    
    if not tensors:
        return torch.empty(0, FUSED_DIM), []
    return torch.stack(tensors, dim=0), valid_hits

def find_video_file(video_name: str) -> Optional[Path]:
    for ext in [".mp4", ".MP4", ".mov", ".MOV", ".avi", ".AVI"]:
        candidate = RAW_VIDEO_DIR / f"{video_name}{ext}"
        if candidate.exists():
            return candidate
    return None

# ── API Models ────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    query: str
    gemini_key: str
    hits: List[Dict[str, Any]]

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/videos")
def get_indexed_videos():
    store = ML_STATE["store"]
    info = store.collection_info()
    return {
        "indexed_videos": list_indexed_videos(),
        "total_segments": info["total_points"]
    }

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    save_path = RAW_VIDEO_DIR / file.filename
    with open(save_path, "wb") as f:
        f.write(await file.read())
    
    try:
        video_embeddings = extract_video_embeddings(str(save_path))
        n_upserted = ML_STATE["store"].upsert_segments(video_embeddings)
        return {"status": "success", "message": f"Indexed {n_upserted} segments."}
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/search")
def search_videos(query: str):
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
        
    query_vector = encode_text_query(query)
    store = ML_STATE["store"]
    fusion_model = ML_STATE["fusion_model"]
    
    hits = store.search(query_vector, top_k=TOP_K_QDRANT)
    if not hits:
        return {"results": []}
        
    segment_matrix, valid_hits = fetch_candidate_tensors(hits)
    if segment_matrix.shape[0] == 0:
        return {"results": []}
        
    query_tensor = torch.tensor(query_vector, dtype=torch.float32)
    top_indices, top_scores = fusion_model.rank_segments(query_tensor, segment_matrix, top_k=TOP_K_RERANK)
    
    results = []
    for rank, (idx, score) in enumerate(zip(top_indices.tolist(), top_scores.tolist())):
        hit = valid_hits[idx]
        vfile = find_video_file(hit["video_name"])
        results.append({
            "rank": rank + 1,
            "video_name": hit["video_name"],
            "filename": vfile.name if vfile else None,
            "timestamp_start": hit["timestamp_start"],
            "timestamp_end": hit["timestamp_end"],
            "attn_score": score,
            "cosine_score": hit["score"],
            "hit_data": hit # used for passing back to Ask endpoint
        })
        
    return {"results": results}

@app.post("/api/ask")
def ask_video(req: AskRequest):
    key = req.gemini_key.strip() if req.gemini_key else ""
    if not key:
        raise HTTPException(status_code=400, detail="Gemini API Key required")
        
    client = genai.Client(api_key=key)
    
    # Extract frames for the top 3 hits
    all_frames = []
    for hit_wrapper in req.hits[:3]:
        hit = hit_wrapper.get("hit_data", hit_wrapper)
        vfile = find_video_file(hit["video_name"])
        if vfile:
            frames = extract_frames_for_segment(
                str(vfile),
                hit["timestamp_start"],
                hit["timestamp_end"] - hit["timestamp_start"],
                num_frames=3
            )
            all_frames.extend(frames)
            
    if not all_frames:
        raise HTTPException(status_code=400, detail="Could not extract frames")

    prompt = f"You are a helpful video assistant. Based on these frames extracted from the most relevant video clips, please answer the user's question.\nQuestion: {req.query}"
    
    def generate():
        try:
            response = client.models.generate_content_stream(
                model='gemini-2.5-flash',
                contents=[prompt] + all_frames
            )
            for chunk in response:
                if chunk.text:
                    yield f"data: {chunk.text}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(str(e))
            yield f"data: [ERROR] {str(e)}\n\n"
            
    return StreamingResponse(generate(), media_type="text/event-stream")

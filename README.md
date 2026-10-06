# 🎬 Multi-Modal Video Retrieval Engine

A production-grade semantic video search system that lets you query your video
library using **natural language** — powered by **CLIP**, **Whisper**,
a custom **Cross-Attention Fusion** PyTorch layer, and a local **Qdrant**
vector database.

---

## Architecture

```
Natural Language Query
        │
        ▼
┌─────────────────────┐
│  CLIP Text Encoder  │  → (1024,) query embedding
└─────────────────────┘
        │
        ▼  Stage 1: Approximate Nearest-Neighbour
┌─────────────────────┐
│  Qdrant (Cosine)    │  → top-20 candidate segments
└─────────────────────┘
        │
        ▼  Stage 2: Exact Re-Ranking
┌─────────────────────────────────────────┐
│  CrossAttentionFusion (PyTorch)         │
│  Attention(Q,K,V) = softmax(QKᵀ/√dk)V  │  → attention scores per segment
└─────────────────────────────────────────┘
        │
        ▼
 st.video() at exact matching timestamp
```

### Indexing Pipeline (per video asset)

```
.mp4 file
   │
   ├── OpenCV frame sampler   → 3 frames / 2-sec segment
   │       └── CLIP ViT-B/32 encoder   → (512,) visual embedding
   │
   └── ffmpeg audio extractor → mono 16kHz waveform / 2-sec segment
           └── Whisper-base encoder    → (512,) audio embedding
                       │
                       ▼
           concat + L2-normalise → (1024,) fused embedding
                       │
                       ▼
           torch.save() → data/processed_tensors/<name>.pt   (cache)
                       │
                       ▼
           Qdrant upsert with payload {video_name, t_start, t_end, seg_idx}
```

---

## Project Structure

```
multimodal-search/
├── data/
│   ├── raw_videos/             ← place source .mp4 files here
│   └── processed_tensors/      ← auto-populated .pt cache files
├── src/
│   ├── __init__.py
│   ├── pipeline.py             ← OpenCV + CLIP + Whisper extraction
│   ├── model.py                ← CrossAttentionFusion nn.Module
│   ├── database.py             ← Qdrant client wrapper
│   └── app.py                  ← Streamlit dashboard
├── local_qdrant_db/            ← auto-created on first run
└── requirements.txt
```

---

## Setup

### Prerequisites

| Tool | Purpose |
|------|---------|
| Python 3.10 or 3.11 | Runtime |
| ffmpeg (in PATH) | Audio extraction from video |
| CUDA 11.8+ (optional) | GPU acceleration |

Install **ffmpeg** on Windows:
```powershell
winget install Gyan.FFmpeg
# or download from https://ffmpeg.org/download.html and add bin/ to PATH
```

### 1 — Create a virtual environment

```powershell
cd "multimodal-search"
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2 — Install dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** The first run downloads ~1 GB of model weights from Hugging Face
> (`openai/clip-vit-base-patch32` and `openai/whisper-base`).
> Subsequent runs use the local Hugging Face cache.

### 3 — Validate the cross-attention module

```powershell
python src/model.py
```

Expected output:
```
CrossAttentionFusion — Dimension Alignment Validation
...
[PASS] All dimension and value assertions satisfied.
```

### 4 — Launch the app

```powershell
streamlit run src/app.py
```

Opens at `http://localhost:8501`.

---

## Usage

### Indexing a video

1. Open the sidebar → **Index New Video**.
2. Upload any `.mp4`, `.mov`, or `.avi` file.
3. Click **🚀 Extract & Index**.

The pipeline segments the video into 2-second clips, runs CLIP + Whisper on each,
and upserts all embeddings into the local Qdrant database. Progress is logged
to the terminal. Subsequent indexing of the same file uses the `.pt` cache and
skips the encoder passes entirely.

### Searching

Type a natural-language description into the search bar and click **Search 🔍**:

```
"a cat sitting on a windowsill"
"someone explaining machine learning concepts"
"explosions and car chases"
"a crowd cheering at a sports event"
```

The engine returns the top-5 matching video segments with:
- Attention weight score (cross-attention stage)
- Cosine similarity score (Qdrant stage)
- Inline `st.video` player starting at the exact timestamp

---

## Module Reference

### `src/pipeline.py`

| Symbol | Description |
|--------|-------------|
| `extract_video_embeddings(path)` | Full ingestion pipeline → `VideoEmbeddings` |
| `list_indexed_videos()` | Returns stems of cached `.pt` files |
| `SegmentEmbedding` | Dataclass: `video_name`, `timestamp_start`, `timestamp_end`, `embedding (1024,)` |
| `VideoEmbeddings` | Container for all segments of one video |

### `src/model.py`

| Symbol | Description |
|--------|-------------|
| `CrossAttentionFusion` | `nn.Module` — scaled dot-product cross-attention |
| `.forward(query, segments)` | Returns `(context (1024,), attn_weights (N,))` |
| `.rank_segments(query, segments, top_k)` | Returns `(top_indices, top_scores)` |

**Attention formula:**
```
Attention(Q, K, V) = softmax( (Q @ Kᵀ) / √d_k ) @ V
```

### `src/database.py`

| Symbol | Description |
|--------|-------------|
| `VectorStore()` | Opens/creates local Qdrant collection |
| `.upsert_segments(video_embeddings)` | Batch-upserts all segments |
| `.search(vector, top_k)` | Cosine ANN search → list of hit dicts |
| `.collection_info()` | Returns stats dict |
| `.get_all_video_names()` | Scrolls collection for unique video names |

### `src/app.py`

Two-stage retrieval orchestration:
1. **Stage 1** — `encode_text_query()` + `VectorStore.search()` (cosine)
2. **Stage 2** — `fetch_candidate_tensors()` + `CrossAttentionFusion.rank_segments()`

---

## Tensor Dimensions Cheat Sheet

| Stage | Shape | Description |
|-------|-------|-------------|
| CLIP visual frames | `(3, 512)` | Per-frame pooler_output |
| Visual embedding | `(512,)` | Mean-pooled + L2-normalised |
| Whisper audio | `(seq, 512)` | Encoder last hidden states |
| Audio embedding | `(512,)` | Mean-pooled + L2-normalised |
| Fused segment | `(1024,)` | `cat([visual, audio])` |
| Text query (fused) | `(1024,)` | `cat([text_512, text_512])` |
| Segment matrix | `(N, 1024)` | All candidate segments |
| Attention weights | `(N,)` | Post-softmax, head-averaged |
| Context vector | `(1024,)` | Attention-weighted output |

---

## Performance Notes

- **Cache**: `.pt` files prevent re-running encoders on already-indexed videos.
- **GPU**: Automatically used if CUDA is available (`torch.cuda.is_available()`).
- **Qdrant HNSW**: `ef=128` for high-recall approximate search at Stage 1.
- **Model freeze**: Both CLIP and Whisper weights are frozen (`requires_grad=False`) — no gradient memory overhead.

---

## License

MIT

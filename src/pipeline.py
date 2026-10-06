"""
pipeline.py — Video Ingestion & Multi-Modal Feature Extraction
==============================================================
1. Decode .mp4 clips with OpenCV into uniform 2-second temporal segments.
2. Sample 3 evenly-spaced visual frames per segment → CLIP visual encoder → (512,) tensor.
3. Extract the corresponding 2-second audio track → Whisper encoder → (512,) tensor.
4. Concatenate visual + audio vectors → (1024,) fused segment embedding.
5. Cache every embedding as a .pt file under data/processed_tensors/.
"""

from __future__ import annotations

import math
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from PIL import Image
from transformers import (
    CLIPProcessor,
    CLIPModel,
    WhisperFeatureExtractor,
    WhisperModel,
)

# ── Path constants ────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_VIDEO_DIR = _PROJECT_ROOT / "data" / "raw_videos"
TENSOR_CACHE_DIR = _PROJECT_ROOT / "data" / "processed_tensors"
TENSOR_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Model identifiers ─────────────────────────────────────────────────────────
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
WHISPER_MODEL_ID = "openai/whisper-base"

# ── Embedding dimensions ──────────────────────────────────────────────────────
VISUAL_DIM = 512
AUDIO_DIM = 512
FUSED_DIM = 1024

# ── Segmentation parameters ───────────────────────────────────────────────────
SEGMENT_DURATION_SEC: float = 2.0
FRAMES_PER_SEGMENT: int = 3
AUDIO_SAMPLE_RATE: int = 16_000


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SegmentEmbedding:
    """One temporal segment + its (1024,) fused embedding."""
    video_name: str
    timestamp_start: float
    timestamp_end: float
    embedding: torch.Tensor  # shape (1024,)

    def to_list(self) -> List[float]:
        return self.embedding.cpu().float().tolist()


@dataclass
class VideoEmbeddings:
    """All segments for a single video asset."""
    video_name: str
    segments: List[SegmentEmbedding] = field(default_factory=list)

    @property
    def num_segments(self) -> int:
        return len(self.segments)


# ─────────────────────────────────────────────────────────────────────────────
# Lazy model singletons
# ─────────────────────────────────────────────────────────────────────────────

class _ModelRegistry:
    _clip_processor: Optional[CLIPProcessor] = None
    _clip_model: Optional[CLIPModel] = None
    _whisper_extractor: Optional[WhisperFeatureExtractor] = None
    _whisper_model: Optional[WhisperModel] = None
    _device: Optional[torch.device] = None

    @classmethod
    def device(cls) -> torch.device:
        if cls._device is None:
            cls._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return cls._device

    @classmethod
    def clip(cls) -> Tuple[CLIPProcessor, CLIPModel]:
        if cls._clip_model is None:
            logger.info(f"Loading CLIP: {CLIP_MODEL_ID}")
            cls._clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
            cls._clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
            cls._clip_model.eval()
            for p in cls._clip_model.parameters():
                p.requires_grad_(False)
            cls._clip_model.to(cls.device())
        return cls._clip_processor, cls._clip_model

    @classmethod
    def whisper(cls) -> Tuple[WhisperFeatureExtractor, WhisperModel]:
        if cls._whisper_model is None:
            logger.info(f"Loading Whisper: {WHISPER_MODEL_ID}")
            cls._whisper_extractor = WhisperFeatureExtractor.from_pretrained(WHISPER_MODEL_ID)
            cls._whisper_model = WhisperModel.from_pretrained(WHISPER_MODEL_ID)
            cls._whisper_model.eval()
            for p in cls._whisper_model.parameters():
                p.requires_grad_(False)
            cls._whisper_model.to(cls.device())
        return cls._whisper_extractor, cls._whisper_model


# ─────────────────────────────────────────────────────────────────────────────
# Audio extraction via ffmpeg subprocess
# ─────────────────────────────────────────────────────────────────────────────

def _extract_audio_segment(
    video_path: str,
    start_sec: float,
    duration_sec: float,
    sample_rate: int = AUDIO_SAMPLE_RATE,
) -> np.ndarray:
    """Slice [start_sec, start_sec+duration_sec] audio from video → mono float32 array."""
    expected_samples = int(sample_rate * duration_sec)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cmd = [
            "ffmpeg", "-loglevel", "error",
            "-ss", str(start_sec), "-t", str(duration_sec),
            "-i", video_path,
            "-ac", "1", "-ar", str(sample_rate),
            "-f", "wav", "-y", tmp_path,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode != 0:
            logger.warning(f"ffmpeg failed @ {start_sec:.1f}s — using zero audio.")
            return np.zeros(expected_samples, dtype=np.float32)

        import soundfile as sf
        audio, sr = sf.read(tmp_path, dtype="float32", always_2d=False)
        if sr != sample_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        if len(audio) < expected_samples:
            audio = np.pad(audio, (0, expected_samples - len(audio)))
        else:
            audio = audio[:expected_samples]
        return audio.astype(np.float32)
    except FileNotFoundError:
        logger.warning("ffmpeg not found — audio embeddings will be zero-valued.")
        return np.zeros(expected_samples, dtype=np.float32)
    except Exception as exc:
        logger.error(f"Audio extraction error: {exc}")
        return np.zeros(expected_samples, dtype=np.float32)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# Per-segment encoding
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _encode_visual_frames(frames: List[np.ndarray]) -> torch.Tensor:
    """BGR frames → CLIP image_features mean → (512,) tensor."""
    processor, model = _ModelRegistry.clip()
    device = _ModelRegistry.device()
    pil_images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
    inputs = processor(images=pil_images, return_tensors="pt").to(device)
    image_features = model.get_image_features(**inputs)  # (n_frames, 512)
    return image_features.mean(dim=0).cpu()              # (512,)


@torch.no_grad()
def _encode_audio_segment(audio: np.ndarray) -> torch.Tensor:
    """Mono float32 audio → Whisper encoder mean-pool → (512,) tensor."""
    extractor, model = _ModelRegistry.whisper()
    device = _ModelRegistry.device()
    inputs = extractor(
        audio, sampling_rate=AUDIO_SAMPLE_RATE, return_tensors="pt"
    ).to(device)
    encoder_out = model.encoder(input_features=inputs["input_features"])
    hidden = encoder_out.last_hidden_state   # (1, seq_len, 512)
    return hidden.squeeze(0).mean(dim=0).cpu()  # (512,)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_frames_for_segment(video_path: str, start_sec: float, duration_sec: float, num_frames: int = 3) -> List[Image.Image]:
    """
    Extracts evenly spaced frames from a specific segment of a video.
    Returns a list of PIL Images (RGB) ready for vision models.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_start = int(start_sec * fps)
    frame_end = min(int((start_sec + duration_sec) * fps), total_frames - 1)

    sample_positions = np.linspace(
        frame_start, frame_end, num=num_frames, endpoint=True, dtype=int
    )

    pil_images = []
    for pos in sample_positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
        ret, frame = cap.read()
        if ret and frame is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_images.append(Image.fromarray(rgb_frame))

    cap.release()
    return pil_images

def extract_video_embeddings(video_path: str) -> VideoEmbeddings:
    """
    Full ingestion pipeline for a single .mp4 file.
    Reads from tensor cache if available, otherwise runs full extraction.
    """
    video_path = Path(video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    video_name = video_path.stem
    cache_path = TENSOR_CACHE_DIR / f"{video_name}.pt"

    # ── Cache hit ─────────────────────────────────────────────────────────────
    if cache_path.exists():
        logger.info(f"[CACHE HIT] {cache_path.name}")
        saved = torch.load(cache_path, map_location="cpu", weights_only=False)
        result = VideoEmbeddings(video_name=video_name)
        for seg_dict in saved:
            result.segments.append(SegmentEmbedding(
                video_name=video_name,
                timestamp_start=seg_dict["timestamp_start"],
                timestamp_end=seg_dict["timestamp_end"],
                embedding=seg_dict["embedding"],
            ))
        logger.info(f"Loaded {result.num_segments} cached segments for '{video_name}'.")
        return result

    # ── Fresh extraction ──────────────────────────────────────────────────────
    logger.info(f"[FRESH] Processing: {video_path.name}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open: {video_path}")

    fps: float = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration_sec: float = total_frames / fps
    num_segments: int = max(1, math.ceil(total_duration_sec / SEGMENT_DURATION_SEC))

    logger.info(
        f"  fps={fps:.2f} | frames={total_frames} | "
        f"duration={total_duration_sec:.1f}s | segments={num_segments}"
    )

    result = VideoEmbeddings(video_name=video_name)

    for seg_idx in range(num_segments):
        t_start = seg_idx * SEGMENT_DURATION_SEC
        t_end = min(t_start + SEGMENT_DURATION_SEC, total_duration_sec)

        frame_start = int(t_start * fps)
        frame_end = min(int(t_end * fps), total_frames - 1)

        sample_positions = np.linspace(
            frame_start, frame_end, num=FRAMES_PER_SEGMENT, endpoint=True, dtype=int
        )

        sampled_frames: List[np.ndarray] = []
        for pos in sample_positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
            ret, frame = cap.read()
            if ret and frame is not None:
                sampled_frames.append(frame)

        if not sampled_frames:
            logger.warning(f"  Segment {seg_idx} has no readable frames — skipping.")
            continue

        # Pad to FRAMES_PER_SEGMENT by duplicating last frame
        while len(sampled_frames) < FRAMES_PER_SEGMENT:
            sampled_frames.append(sampled_frames[-1].copy())

        visual_embed = _encode_visual_frames(sampled_frames)  # (512,)
        audio_array = _extract_audio_segment(str(video_path), t_start, SEGMENT_DURATION_SEC)
        audio_embed = _encode_audio_segment(audio_array)      # (512,)

        fused_embed = torch.cat([
            F.normalize(visual_embed, dim=0),
            F.normalize(audio_embed, dim=0),
        ], dim=0)  # (1024,)

        result.segments.append(SegmentEmbedding(
            video_name=video_name,
            timestamp_start=round(t_start, 3),
            timestamp_end=round(t_end, 3),
            embedding=fused_embed,
        ))
        logger.info(
            f"  Seg {seg_idx + 1}/{num_segments} [{t_start:.1f}s→{t_end:.1f}s] "
            f"shape={tuple(fused_embed.shape)}"
        )

    cap.release()

    if result.num_segments == 0:
        raise RuntimeError(
            f"No segments extracted from '{video_path.name}'. "
            "Verify the file is a valid, non-corrupted MP4."
        )

    # ── Persist cache ─────────────────────────────────────────────────────────
    cache_payload = [
        {
            "timestamp_start": seg.timestamp_start,
            "timestamp_end": seg.timestamp_end,
            "embedding": seg.embedding,
        }
        for seg in result.segments
    ]
    torch.save(cache_payload, cache_path)
    logger.info(f"Cached {result.num_segments} tensors → {cache_path}")
    return result


def list_indexed_videos() -> List[str]:
    """Return stems of all videos that have cached .pt files."""
    return [p.stem for p in TENSOR_CACHE_DIR.glob("*.pt")]

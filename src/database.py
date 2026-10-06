"""
database.py — Local Qdrant Vector Store Integration
====================================================
Responsibilities
----------------
1. Instantiate a local (on-disk) Qdrant client backed by a filesystem path.
2. Auto-create the collection on first run using Cosine distance and
   vector size = FUSED_DIM (1024).
3. Upsert SegmentEmbedding objects with rich payload metadata.
4. Expose a semantic search method returning ranked hit results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

# ── Path constants ────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_QDRANT_PATH = str(_PROJECT_ROOT / "local_qdrant_db")

# ── Collection configuration ──────────────────────────────────────────────────
COLLECTION_NAME = "multimodal_segments"
VECTOR_SIZE = 1024         # must match FUSED_DIM in pipeline.py
DISTANCE_METRIC = qmodels.Distance.COSINE


# ─────────────────────────────────────────────────────────────────────────────
# VectorStore class
# ─────────────────────────────────────────────────────────────────────────────

class VectorStore:
    """
    Thin wrapper around a local Qdrant client for multi-modal segment storage
    and semantic retrieval.

    Usage
    -----
        store = VectorStore()
        store.upsert_segments(video_embeddings)
        hits = store.search(query_vector, top_k=10)
    """

    def __init__(
        self,
        path: str = LOCAL_QDRANT_PATH,
        collection_name: str = COLLECTION_NAME,
        vector_size: int = VECTOR_SIZE,
    ) -> None:
        self.collection_name = collection_name
        self.vector_size = vector_size

        logger.info(f"Connecting to local Qdrant at: {path}")
        self._client = QdrantClient(path=path)
        self._ensure_collection()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _ensure_collection(self) -> None:
        """Create the vector collection if it does not already exist."""
        existing = [c.name for c in self._client.get_collections().collections]
        if self.collection_name in existing:
            info = self._client.get_collection(self.collection_name)
            stored_size = info.config.params.vectors.size
            if stored_size != self.vector_size:
                raise RuntimeError(
                    f"Collection '{self.collection_name}' already exists with "
                    f"vector size {stored_size}, but this session expects "
                    f"{self.vector_size}. Delete the 'local_qdrant_db' directory "
                    "and re-index to resolve the mismatch."
                )
            logger.info(
                f"Collection '{self.collection_name}' already exists "
                f"(vector_size={stored_size}) — reusing."
            )
            return

        logger.info(
            f"Creating collection '{self.collection_name}' "
            f"(size={self.vector_size}, distance=Cosine)."
        )
        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qmodels.VectorParams(
                size=self.vector_size,
                distance=DISTANCE_METRIC,
            ),
        )
        logger.info("Collection created successfully.")

    @staticmethod
    def _make_point_id(video_name: str, seg_idx: int) -> int:
        """
        Deterministic integer point ID derived from video name hash + segment index.
        Qdrant requires integer or UUID point IDs for local collections.
        """
        # Shift the hash to stay in positive int64 range and add seg_idx offset
        base = abs(hash(video_name)) % (10 ** 15)
        return base + seg_idx

    # ── Public write API ──────────────────────────────────────────────────────

    def upsert_segments(self, video_embeddings: Any) -> int:
        """
        Insert or update all segments from a VideoEmbeddings object.

        Parameters
        ----------
        video_embeddings : VideoEmbeddings
            Object returned by pipeline.extract_video_embeddings().

        Returns
        -------
        int
            Number of points successfully upserted.
        """
        if not video_embeddings.segments:
            logger.warning(
                f"No segments to upsert for '{video_embeddings.video_name}'."
            )
            return 0

        points: List[qmodels.PointStruct] = []
        for seg_idx, seg in enumerate(video_embeddings.segments):
            point_id = self._make_point_id(video_embeddings.video_name, seg_idx)
            vector = seg.to_list()

            if len(vector) != self.vector_size:
                logger.error(
                    f"Segment {seg_idx} of '{video_embeddings.video_name}' has "
                    f"vector length {len(vector)}, expected {self.vector_size} — skipping."
                )
                continue

            payload: Dict[str, Any] = {
                "video_name": seg.video_name,
                "timestamp_start": seg.timestamp_start,
                "timestamp_end": seg.timestamp_end,
                "segment_index": seg_idx,
            }
            points.append(
                qmodels.PointStruct(id=point_id, vector=vector, payload=payload)
            )

        if not points:
            logger.error("All segments were skipped due to dimension mismatches.")
            return 0

        # Batch upsert (Qdrant handles idempotency via upsert semantics)
        self._client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
        logger.info(
            f"Upserted {len(points)} segments for '{video_embeddings.video_name}' "
            f"into '{self.collection_name}'."
        )
        return len(points)

    def delete_video(self, video_name: str, num_segments: int) -> None:
        """Remove all segments belonging to a specific video from the collection."""
        ids_to_delete = [
            self._make_point_id(video_name, i) for i in range(num_segments)
        ]
        self._client.delete(
            collection_name=self.collection_name,
            points_selector=qmodels.PointIdsList(points=ids_to_delete),
            wait=True,
        )
        logger.info(
            f"Deleted {len(ids_to_delete)} points for video '{video_name}'."
        )

    # ── Public read API ───────────────────────────────────────────────────────

    def search(
        self,
        query_vector: List[float],
        top_k: int = 20,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Stage-1 cosine similarity search against stored segment embeddings.

        Parameters
        ----------
        query_vector : List[float]
            The text query embedding as a Python list of floats.
        top_k : int
            Maximum number of candidates to return.
        score_threshold : float | None
            If provided, only segments with cosine score >= this value are returned.

        Returns
        -------
        List[Dict[str, Any]]
            Each dict contains keys: id, score, video_name, timestamp_start,
            timestamp_end, segment_index.
        """
        if len(query_vector) != self.vector_size:
            raise ValueError(
                f"Query vector length {len(query_vector)} does not match "
                f"collection vector size {self.vector_size}."
            )

        search_params = qmodels.SearchParams(hnsw_ef=128, exact=False)

        results = self._client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            search_params=search_params,
            with_payload=True,
        )

        hits: List[Dict[str, Any]] = []
        for hit in results:
            payload = hit.payload or {}
            hits.append({
                "id": hit.id,
                "score": round(hit.score, 6),
                "video_name": payload.get("video_name", "unknown"),
                "timestamp_start": payload.get("timestamp_start", 0.0),
                "timestamp_end": payload.get("timestamp_end", 0.0),
                "segment_index": payload.get("segment_index", -1),
            })

        logger.debug(f"Qdrant search returned {len(hits)} candidates.")
        return hits

    def collection_info(self) -> Dict[str, Any]:
        """Return basic collection statistics as a dict."""
        info = self._client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "total_points": info.points_count,
            "vector_size": info.config.params.vectors.size,
            "distance": str(info.config.params.vectors.distance),
            "status": str(info.status),
        }

    def get_all_video_names(self) -> List[str]:
        """Scroll through the collection and return unique video names in the payload."""
        names: set = set()
        offset = None
        while True:
            records, next_offset = self._client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for rec in records:
                if rec.payload:
                    names.add(rec.payload.get("video_name", ""))
            if next_offset is None:
                break
            offset = next_offset
        names.discard("")
        return sorted(names)

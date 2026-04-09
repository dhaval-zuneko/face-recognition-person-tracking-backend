"""
RunPod Serverless Worker — Shyam Steels
========================================
Mirrors the inference logic already in services/colab_client.py
but runs on RunPod GPU instead of locally.

Payload (input):
{
  "frames_b64": ["<base64 jpeg>", ...],     # list of frames to process
  "face_db": {                               # snapshot of all embeddings from SQLite
    "VIS001": [{"embedding": [...], "type": "visitor"}, ...],
    "EMP001": [{"embedding": [...], "type": "employee"}, ...],
    ...
  },
  "camera_id": "floor2",
  "threshold": 0.6                           # optional, default 0.6
}

Response (output):
{
  "results": [
    {
      "frame_index": 0,
      "detections": [
        {
          "label": "employee",
          "person_id": "EMP001",
          "confidence": 0.87,
          "bbox": [x, y, w, h]
        }
      ]
    },
    ...
  ]
}
"""

import base64
import json
import logging

import cv2
import numpy as np
import runpod
from deepface import DeepFace

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("runpod_worker")

# Pre-warm DeepFace on cold start (downloads Facenet weights once)
logger.info("Pre-warming DeepFace Facenet...")
try:
    _dummy = np.zeros((160, 160, 3), dtype=np.uint8)
    DeepFace.represent(_dummy, model_name="Facenet", enforce_detection=False)
    logger.info("DeepFace ready.")
except Exception as e:
    logger.warning(f"Pre-warm failed (will retry on first request): {e}")


# ─── helpers ──────────────────────────────────────────────────────────────────

def _b64_to_frame(b64_str: str) -> np.ndarray:
    img_bytes = base64.b64decode(b64_str)
    arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _cosine_similarity(a, b) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def _run_inference_on_frame(frame: np.ndarray, face_db: dict, threshold: float) -> list[dict]:
    """
    Exact same logic as colab_client.detect_faces(), but self-contained.
    Returns list of detection dicts.
    """
    try:
        faces = DeepFace.represent(
            img_path=frame,
            model_name="Facenet",
            enforce_detection=False,
        )
    except Exception as e:
        logger.warning(f"DeepFace.represent error: {e}")
        return []

    detections = []
    for face_data in faces:
        incoming = np.array(face_data["embedding"])
        region   = face_data.get("facial_area", {})

        best_id, best_type, best_score = None, "unknown", 0.0

        for person_id, emb_list in face_db.items():
            for entry in emb_list:
                score = _cosine_similarity(incoming, entry["embedding"])
                if score > best_score:
                    best_score = score
                    best_id    = person_id
                    best_type  = entry["type"]

        label     = best_type if best_score > threshold else "unknown"
        person_id = best_id   if best_score > threshold else None

        detections.append({
            "label":      label,
            "person_id":  person_id,
            "confidence": round(best_score, 4),
            "bbox": [
                region.get("x", 0), region.get("y", 0),
                region.get("w", 0), region.get("h", 0),
            ],
        })

    return detections


# ─── RunPod handler ───────────────────────────────────────────────────────────

def handler(job: dict) -> dict:
    inp       = job.get("input", {})
    frames_b64 = inp.get("frames_b64", [])
    face_db    = inp.get("face_db", {})
    threshold  = float(inp.get("threshold", 0.6))
    camera_id  = inp.get("camera_id", "unknown")

    if not frames_b64:
        return {"error": "No frames provided", "results": []}

    logger.info(f"[{camera_id}] Processing {len(frames_b64)} frames, {len(face_db)} people in DB")

    results = []
    for idx, b64 in enumerate(frames_b64):
        frame = _b64_to_frame(b64)
        if frame is None:
            results.append({"frame_index": idx, "detections": [], "error": "decode_failed"})
            continue

        detections = _run_inference_on_frame(frame, face_db, threshold)
        results.append({"frame_index": idx, "detections": detections})

    total_matches = sum(
        1 for r in results
        for d in r["detections"]
        if d["label"] != "unknown"
    )
    logger.info(f"[{camera_id}] Done — {total_matches} matches across {len(results)} frames")

    return {"results": results, "camera_id": camera_id}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

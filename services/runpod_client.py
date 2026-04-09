"""
services/runpod_client.py
==========================
Drop-in replacement for colab_client.py.
Sends frames to RunPod serverless endpoint instead of running DeepFace locally.

Set these env vars (add to .env):
    RUNPOD_API_KEY      = your RunPod API key
    RUNPOD_ENDPOINT_ID  = your endpoint ID (from RunPod dashboard)

The API contract (enroll_face / detect_faces) is identical to colab_client.py
so detection.py and visitor.py routes need zero changes.
"""

import os
import base64
import json
import time
import logging
import httpx

from services import face_db as fdb

logger = logging.getLogger("runpod_client")

RUNPOD_API_KEY     = os.getenv("RUNPOD_API_KEY", "")
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID", "")
RUNPOD_BASE_URL    = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}"
RUNPOD_TIMEOUT     = int(os.getenv("RUNPOD_TIMEOUT", "120"))
MATCH_THRESHOLD    = float(os.getenv("MATCH_THRESHOLD", "0.6"))

_HEADERS = {
    "Authorization": f"Bearer {RUNPOD_API_KEY}",
    "Content-Type": "application/json",
}


# ─── internal helpers ─────────────────────────────────────────────────────────

def _frame_to_b64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


async def _call_runpod(payload: dict) -> dict:
    if not RUNPOD_API_KEY or not RUNPOD_ENDPOINT_ID:
        raise RuntimeError(
            "RunPod not configured. Set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID."
        )

    async with httpx.AsyncClient(timeout=RUNPOD_TIMEOUT) as client:
        resp = await client.post(
            f"{RUNPOD_BASE_URL}/runsync",
            headers=_HEADERS,
            json={"input": payload},
        )
        resp.raise_for_status()
        data = resp.json()

        # If worker was cold, runsync returns IN_QUEUE — poll until done
        if data.get("status") in ("IN_QUEUE", "IN_PROGRESS"):
            job_id = data["id"]
            logger.info(f"RunPod job {job_id} queued, polling...")
            data = await _poll(client, job_id)

        return data.get("output", data)


async def _poll(client: httpx.AsyncClient, job_id: str) -> dict:
    import asyncio
    deadline = time.time() + RUNPOD_TIMEOUT
    while time.time() < deadline:
        await asyncio.sleep(2)
        r = await client.get(f"{RUNPOD_BASE_URL}/status/{job_id}", headers=_HEADERS)
        d = r.json()
        if d.get("status") == "COMPLETED":
            return d
        if d.get("status") == "FAILED":
            raise RuntimeError(f"RunPod job failed: {d}")
    raise TimeoutError(f"RunPod job {job_id} timed out after {RUNPOD_TIMEOUT}s")


# ─── public API (same contract as colab_client.py) ────────────────────────────

async def enroll_face(image_bytes: bytes, filename: str, person_id: str, person_type: str) -> bool:
    """
    Enroll a face: extract embedding via DeepFace locally (fast, no GPU needed)
    and store it in the SQLite DB.  RunPod is not needed for enrollment.
    """
    import numpy as np
    import cv2
    from deepface import DeepFace

    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    try:
        embedding = DeepFace.represent(
            img_path=frame,
            model_name="Facenet",
            enforce_detection=False,
        )[0]["embedding"]

        fdb.add_embedding(person_id, embedding, person_type)
        logger.info(f"Enrolled {person_type} {person_id}")
        return True

    except Exception as e:
        logger.error(f"Enroll error for {person_id}: {e}")
        return False


async def detect_faces(frame_bytes: bytes, filename: str, camera_id: str) -> list[dict]:
    """
    Send ONE frame to RunPod for face detection + recognition.
    Returns list of detection dicts (same shape as colab_client.py).
    """
    # Pass the full face DB snapshot so RunPod can match without a DB connection
    all_embeddings = fdb.get_all_embeddings()

    payload = {
        "camera_id":  camera_id,
        "frames_b64": [_frame_to_b64(frame_bytes)],
        "face_db":    all_embeddings,
        "threshold":  MATCH_THRESHOLD,
    }

    try:
        result = await _call_runpod(payload)
        frame_results = result.get("results", [])
        if frame_results:
            return frame_results[0].get("detections", [])
        return []

    except Exception as e:
        logger.error(f"[{camera_id}] RunPod detect_faces error: {e}")
        return []


async def detect_faces_batch(frames_bytes: list[bytes], camera_id: str) -> list[list[dict]]:
    """
    Send MULTIPLE frames in one RunPod call (used by the video endpoint).
    Returns list of detection lists, one per frame.
    """
    all_embeddings = fdb.get_all_embeddings()

    payload = {
        "camera_id":  camera_id,
        "frames_b64": [_frame_to_b64(fb) for fb in frames_bytes],
        "face_db":    all_embeddings,
        "threshold":  MATCH_THRESHOLD,
    }

    try:
        result = await _call_runpod(payload)
        frame_results = result.get("results", [])
        return [fr.get("detections", []) for fr in frame_results]

    except Exception as e:
        logger.error(f"[{camera_id}] RunPod detect_faces_batch error: {e}")
        return [[] for _ in frames_bytes]

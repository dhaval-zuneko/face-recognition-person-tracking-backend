"""
Detection Route  (updated)
--------------------------
POST /api/camera/{camera_id}/frame

Now accepts BOTH:
  • Single image frame  (jpg/png)  — same behaviour as before
  • Recorded video file (mp4/avi)  — extracts frames, batches to RunPod

Query param:
  ?sample_every=5   process 1 of every N frames (default 5, min 1)

Response shape is the same for both:
{
  "camera_id": "floor2",
  "detections": [...],          # all detections across all processed frames
  "frames_processed": 12,       # only present for video
  "source": "image" | "video"
}
"""

import base64
import io
import os
import tempfile
import time
import logging
from multiprocessing import Process
from camera_worker import run_camera
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, Query
from services.ws_manager import manager
from services import face_db
import cv2
import os
from fastapi import APIRouter, UploadFile, File
from services import colab_client  # or local_inference
router = APIRouter()
logger = logging.getLogger("detection")

# ── swap which client to use via env var ──────────────────────────────────────
# Set USE_RUNPOD=1 in your .env to route through RunPod.
# Leave unset (or =0) to keep using the local DeepFace pipeline.


UPLOAD_DIR = "uploaded_videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)
running_processes = {}

_USE_RUNPOD = os.getenv("USE_RUNPOD", "0") == "1"

if _USE_RUNPOD:
    from services import runpod_client as inference_client
    logger.info("Detection route → RunPod inference")
else:
    from services import colab_client as inference_client
    logger.info("Detection route → local DeepFace inference")


router = APIRouter()

CAMERA_ROLES: dict[str, str] = {
    "entry":  "entry",
    "floor1": "floor",
    "floor2": "floor",
    "floor3": "floor",
    "zone_a": "floor",
}

_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# ─── helpers ──────────────────────────────────────────────────────────────────

def _draw_detections(frame, detections):
    for det in detections:
        x, y, w, h = det.get("box", [0, 0, 0, 0])
        label = det.get("label", "unknown")
        score = det.get("score", 0)

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw text
        text = f"{label} ({score:.2f})"
        cv2.putText(frame, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

def _is_video(file: UploadFile) -> bool:
    ct = (file.content_type or "").lower()
    name = (file.filename or "").lower()
    ext = os.path.splitext(name)[1]
    return "video" in ct or ext in _VIDEO_EXTENSIONS


def _extract_frames(video_bytes: bytes, sample_every: int) -> list[tuple[int, bytes]]:
    """
    Write video to a temp file, extract frames with OpenCV.
    Returns list of (frame_index, jpeg_bytes) for sampled frames.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    frames = []
    try:
        cap = cv2.VideoCapture(tmp_path)
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % sample_every == 0:
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok:
                    frames.append((idx, buf.tobytes()))
            idx += 1
        cap.release()
    finally:
        os.unlink(tmp_path)

    return frames


def _enrich(det: dict) -> dict:
    """Add name from local DB to a detection dict."""
    label     = det.get("label", "unknown")
    person_id = det.get("person_id")

    if label == "employee" and person_id:
        emp = face_db.get_employee(person_id)
        det["name"]   = emp["name"] if emp else person_id
        det["emp_id"] = person_id

    elif label == "visitor" and person_id:
        vis = face_db.get_visitor(person_id)
        det["visitor_id"] = person_id
        det["name"]       = vis["name"] if vis else person_id

    return det


# ─── route ────────────────────────────────────────────────────────────────────

@router.post("/camera/{camera_id}/frame")
async def process_frame(
    camera_id: str,
    file: UploadFile = File(...),
    sample_every: int = Query(5, ge=1, description="For video: process 1 of every N frames"),
):
    """
    Accept a JPEG/PNG frame OR a recorded video (mp4/avi) from a camera.
    Runs face detection + recognition and broadcasts results to Receptionist UI.
    """
    file_bytes  = await file.read()
    camera_role = CAMERA_ROLES.get(camera_id, "floor")

    # ── VIDEO path ─────────────────────────────────────────────────────────────
    if _is_video(file):
        return await _process_video(
            file_bytes, camera_id, camera_role, sample_every, file.filename
        )

    # ── SINGLE IMAGE path (original behaviour, untouched) ──────────────────────
    return await _process_image(
        file_bytes, camera_id, camera_role, file.filename
    )

@router.post("/camera/{camera_id}/start")
async def start_camera(camera_id: str, file: UploadFile = File(...)):

    # Save video
    file_path = os.path.join(UPLOAD_DIR, f"{camera_id}.mp4")
    
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Start worker process
    p = Process(target=run_camera, args=(camera_id, file_path))
    p.start()

    running_processes[camera_id] = p

    return {
        "status": "started",
        "camera_id": camera_id,
        "file": file_path
    }

# # 🎥 CAMERA 1
# @router.post("/camera1/frame")
# async def camera1_frame(file: UploadFile = File(...)):
#     contents = await file.read()

#     detections = await colab_client.detect_faces(
#         frame_bytes=contents,
#         filename=file.filename,
#         camera_id="camera1"
#     )

#     print("📷 Camera 1 processed")

#     return {
#         "camera": "camera1",
#         "detections": detections
#     }


# 🎥 CAMERA 2
# @router.post("/camera2/frame")
# async def camera2_frame(file: UploadFile = File(...)):
#     contents = await file.read()

#     detections = await colab_client.detect_faces(
#         frame_bytes=contents,
#         filename=file.filename,
#         camera_id="camera2"
#     )

#     print("📷 Camera 2 processed")

#     return {
#         "camera": "camera2",
#         "detections": detections
#     }

# ─── image (original logic, unchanged) ────────────────────────────────────────

async def _process_image(
    frame_bytes: bytes, camera_id: str, camera_role: str, filename: str
) -> dict:
    raw_detections = await inference_client.detect_faces(
        frame_bytes=frame_bytes,
        filename=filename or f"{camera_id}.jpg",
        camera_id=camera_id,
    )

    enriched = [_enrich(d) for d in raw_detections]

    await manager.broadcast_detection(
        camera_id=camera_id,
        camera_role=camera_role,
        detections=enriched,
    )

    if camera_role == "entry":
        for det in enriched:
            if det.get("label") == "unknown":
                await manager.notify_unknown(
                    camera_id=camera_id,
                    face_crop_b64=det.get("face_crop_b64"),
                )
                break

    return {"camera_id": camera_id, "detections": enriched, "source": "image"}


# ─── video ─────────────────────────────────────────────────────────────────────

async def _process_video(
    video_bytes: bytes, camera_id: str, camera_role: str, sample_every: int, filename: str
) -> dict:
    logger.info(
        f"[{camera_id}] Video upload: {len(video_bytes)//1024} KB, "
        f"sample_every={sample_every}"
    )

    # 1. Extract sampled frames
    sampled_frames = _extract_frames(video_bytes, sample_every)
    if not sampled_frames:
        return {
            "camera_id": camera_id,
            "detections": [],
            "frames_processed": 0,
            "source": "video",
            "error": "No frames extracted — check video format",
        }

    logger.info(f"[{camera_id}] Extracted {len(sampled_frames)} sampled frames")

    # 2. Batch inference on RunPod (one round-trip for all frames)
    if _USE_RUNPOD and hasattr(inference_client, "detect_faces_batch"):
        frame_bytes_list = [fb for _, fb in sampled_frames]
        per_frame_detections = await inference_client.detect_faces_batch(
            frames_bytes=frame_bytes_list,
            camera_id=camera_id,
        )
    else:
        # Fallback: call per-frame (works with local client too)
        per_frame_detections = []
        for _, fb in sampled_frames:
            dets = await inference_client.detect_faces(
                frame_bytes=fb,
                filename=f"{camera_id}_frame.jpg",
                camera_id=camera_id,
            )
            per_frame_detections.append(dets)
    output_dir = "processed_videos"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"processed_{int(time.time())}.mp4")

    # Get first frame for size
    _, first_fb = sampled_frames[0]
    first_frame = cv2.imdecode(np.frombuffer(first_fb, np.uint8), cv2.IMREAD_COLOR)
    h, w, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 10, (w, h))

    # Loop using YOUR structure
    for (frame_idx, fb), detections in zip(sampled_frames, per_frame_detections):
        frame = cv2.imdecode(np.frombuffer(fb, np.uint8), cv2.IMREAD_COLOR)

        for det in detections:
            bbox = det.get("box") or det.get("bbox") or [0, 0, 0, 0]
            x, y, w_box, h_box = bbox

            label = det.get("label", "unknown")
            score = det.get("score", 0)

            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({score:.2f})",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2)

        out.write(frame)

    out.release()

    print(f"✅ Processed video saved at: {output_path}")

    # 3. Enrich + flatten all detections with frame metadata
    all_enriched = []
    for (frame_idx, _), dets in zip(sampled_frames, per_frame_detections):
        for det in dets:
            enriched = _enrich(det)
            enriched["frame_index"] = frame_idx
            all_enriched.append(enriched)

    # 4. Broadcast the last frame's detections to the UI (most recent position)
    if per_frame_detections and per_frame_detections[-1]:
        last_enriched = [_enrich(d) for d in per_frame_detections[-1]]
        await manager.broadcast_detection(
            camera_id=camera_id,
            camera_role=camera_role,
            detections=last_enriched,
        )

    # 5. Unknown-at-entry check (any frame)
    if camera_role == "entry":
        for det in all_enriched:
            if det.get("label") == "unknown":
                await manager.notify_unknown(
                    camera_id=camera_id,
                    face_crop_b64=det.get("face_crop_b64"),
                )
                break

    logger.info(
        f"[{camera_id}] Video done — {len(sampled_frames)} frames, "
        f"{len(all_enriched)} detections"
    )

    return {
        "camera_id":        camera_id,
        "detections":       all_enriched,
        "frames_processed": len(sampled_frames),
        "source":           "video",
    }

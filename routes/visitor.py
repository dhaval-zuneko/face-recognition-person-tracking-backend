"""
Visitor Route
-------------
Two endpoints:

1. POST /api/visitors/register
   Called by the Receptionist UI after they click the "Register" button
   for an unknown face.  Accepts a photo, assigns a Visitor ID, enrolls
   the face in Colab, and broadcasts the new visitor to all WS clients.

2. GET /api/visitors
   Returns all registered visitors (for the presence dashboard).
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from services import face_db, colab_client
from services.ws_manager import manager

router = APIRouter()


@router.post("/visitors/register")
async def register_visitor(
    name:             str        = Form(...),
    host:             str        = Form(...),
    permitted_floors: str        = Form(...),   # comma-separated, e.g. "1,2"
    photo:            UploadFile = File(...),
):
    """
    Register a new visitor and enroll their face.

    Flow:
      Receptionist UI → uploads photo here
      → backend creates Visitor ID (V001, V002, …)
      → sends photo to Colab for face enrollment
      → broadcasts "visitor_registered" event over WebSocket
      → returns { visitor_id, name, status }
    """
    # 1. Create visitor record in local DB
    floors = [f.strip() for f in permitted_floors.split(",")]
    visitor = face_db.create_visitor(name=name, host=host, permitted_floors=floors)
    visitor_id = visitor["visitor_id"]

    # 2. Enroll face in Colab
    image_bytes = await photo.read()
    enrolled = await colab_client.enroll_face(
        image_bytes=image_bytes,
        filename=photo.filename or f"{visitor_id}.jpg",
        person_id=visitor_id,
        person_type="visitor",
    )

    if enrolled:
        face_db.mark_visitor_enrolled(visitor_id)
    else:
        # Colab unavailable — visitor is registered but not yet enrolled.
        # Future frames won't match until Colab is back; that's acceptable for POC.
        print(f"[Visitor] Warning: could not enroll {visitor_id} in Colab (will retry on next frame)")

    # 3. Broadcast to Receptionist UI
    await manager.notify_visitor_registered(visitor_id=visitor_id, name=name)

    return {
        "status": "registered",
        "visitor_id": visitor_id,
        "name": name,
        "enrolled_in_colab": enrolled,
    }


@router.get("/visitors")
def list_visitors():
    return {"visitors": face_db.list_visitors()}

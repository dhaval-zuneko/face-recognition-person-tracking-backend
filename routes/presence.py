"""
Presence Route
--------------
Returns who is currently active/visible across all cameras.
Used by the live dashboard on the Receptionist UI.

For the POC this is a simple in-memory last-seen store.
In production this would hit the location_events table.
"""

from fastapi import APIRouter
from datetime import datetime

router = APIRouter()

# In-memory last-seen store
# { person_id: { "label", "name", "camera_id", "last_seen" } }
_last_seen: dict[str, dict] = {}


def update_presence(detections: list[dict], camera_id: str):
    """Called by the detection pipeline to keep this store fresh."""
    for det in detections:
        label = det.get("label")
        if label == "unknown":
            continue
        person_id = det.get("emp_id") or det.get("visitor_id")
        if not person_id:
            continue
        _last_seen[person_id] = {
            "label":     label,
            "name":      det.get("name", person_id),
            "person_id": person_id,
            "camera_id": camera_id,
            "last_seen": datetime.now().isoformat(),
        }


@router.get("/presence")
def get_presence():
    """Return all currently tracked persons, sorted by last_seen desc."""
    persons = sorted(
        _last_seen.values(),
        key=lambda x: x["last_seen"],
        reverse=True,
    )
    return {"active": persons, "count": len(persons)}

"""
WebSocket Connection Manager
Broadcasts detection events to all connected receptionist UI clients.
"""

import json
from typing import List
from fastapi import WebSocket


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"[WS] Client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"[WS] Client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Send a JSON message to all connected clients."""
        payload = json.dumps(message)
        dead = []
        for connection in self.active_connections:
            try:
                await connection.send_text(payload)
            except Exception:
                dead.append(connection)
        for c in dead:
            self.disconnect(c)

    async def broadcast_detection(
        self,
        camera_id: str,
        camera_role: str,  # "entry" | "floor"
        detections: list,
    ):
        """
        Broadcast a structured detection event.

        Each detection item should be:
          {
            "label": "employee" | "visitor" | "unknown",
            "name": str | None,
            "emp_id": str | None,
            "visitor_id": str | None,
            "confidence": float,
            "face_crop_b64": str | None   # only for unknowns at entry camera
          }
        """
        await self.broadcast({
            "event": "detection",
            "camera_id": camera_id,
            "camera_role": camera_role,
            "detections": detections,
        })

    async def notify_visitor_registered(self, visitor_id: str, name: str):
        await self.broadcast({
            "event": "visitor_registered",
            "visitor_id": visitor_id,
            "name": name,
        })

    async def notify_unknown(self, camera_id: str, face_crop_b64: str | None):
        """
        Sent specifically to the receptionist when an unknown face hits
        the entry camera — prompts them to register the visitor.
        """
        await self.broadcast({
            "event": "unknown_at_entry",
            "camera_id": camera_id,
            "face_crop_b64": face_crop_b64,
        })


# Singleton — imported by both routes and startup
manager = ConnectionManager()

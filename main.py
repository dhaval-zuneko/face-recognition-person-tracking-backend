# from fastapi import FastAPI, UploadFile, File, Form
# from fastapi.middleware.cors import CORSMiddleware
# import requests
# import uuid

# app = FastAPI()

# # Allow frontend access
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # 👉 Replace with your Colab ngrok URL
# COLAB_URL = "https://your-colab-ngrok-url.ngrok.io"

# # -----------------------------
# # 1. GET DETECTIONS (Frontend Polling)
# # -----------------------------
# @app.get("/api/detections")
# async def get_detections():
#     try:
#         res = requests.get(f"{COLAB_URL}/detect")
#         return res.json()
#     except:
#         return {"detections": {}}


# # -----------------------------
# # 2. REGISTER VISITOR
# # -----------------------------
# @app.post("/api/visitors/register")
# async def register_visitor(
#     visitor_id: str = Form(...),
#     name: str = Form(...),
#     host: str = Form(...),
#     permitted_floor: str = Form(...),
#     photo: UploadFile = File(None)
# ):
#     print("Visitor Registered:", visitor_id, name)

#     # Send to Colab (optional for face encoding)
#     if photo:
#         files = {"file": (photo.filename, await photo.read())}
#         data = {"visitor_id": visitor_id}

#         try:
#             requests.post(f"{COLAB_URL}/register", files=files, data=data)
#         except:
#             pass

#     return {"status": "ok"}

"""
Shyam Steels — Visitor & Employee Tracking Backend
====================================================
FastAPI application that:

  1. Accepts camera frames via HTTP POST
  2. Forwards them to the Colab face-recognition server
  3. Enriches detections with employee/visitor metadata
  4. Pushes real-time events to the Receptionist UI over WebSocket
  5. Handles visitor registration (unknown → Visitor ID + Colab enrollment)

Run with:
    uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from routes import detection, visitor, employee, presence
from services.ws_manager import manager
from services.database import init_db
init_db()   # creates tables if they don't exist

app = FastAPI(title="Shyam Steels Tracking API", version="1.0.0")

# ── CORS (allow the React frontend on any port during dev) ────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(detection.router, prefix="/api", tags=["Detection"])
app.include_router(visitor.router,   prefix="/api", tags=["Visitors"])
app.include_router(employee.router,  prefix="/api", tags=["Employees"])
app.include_router(presence.router,  prefix="/api", tags=["Presence"])


# ── WebSocket endpoint ─────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Receptionist UI connects here to receive real-time events.

    Events pushed by the server:
      - detection        → new faces seen on any camera
      - unknown_at_entry → unknown face at entry camera (prompts registration)
      - visitor_registered → new visitor successfully registered
    """
    await manager.connect(websocket)
    try:
        while True:
            # Keep the connection alive; we only push from server → client.
            # If the client sends anything (e.g. a ping), just ignore it.
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


# ── Dev entrypoint ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

# """
# LOCAL inference (NO COLAB)
# """

# from services.local_inference import detect_faces_local
# from services.face_db import face_db
# import numpy as np
# import cv2


# async def enroll_face(image_bytes, filename, person_id, person_type):
#     from deepface import DeepFace
#     import numpy as np
#     import cv2

#     np_arr = np.frombuffer(image_bytes, np.uint8)
#     frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

#     try:
#         embedding = DeepFace.represent(
#             img_path=frame,
#             model_name="Facenet",
#             enforce_detection=False
#         )[0]["embedding"]

#         if person_id not in face_db:
#             face_db[person_id] = []

#         face_db[person_id].append({
#             "embedding": embedding,
#             "type": person_type
#         })

#         return True

#     except Exception as e:
#         print("Enroll error:", e)
#         return False


# async def detect_faces(frame_bytes, filename, camera_id):
#     np_arr = np.frombuffer(frame_bytes, np.uint8)
#     frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

#     detections = detect_faces_local(frame, face_db)

#     return detections

"""
LOCAL inference (NO COLAB)
"""

from deepface import DeepFace
from services import face_db as fdb
import numpy as np
import cv2


async def enroll_face(image_bytes, filename, person_id, person_type):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    try:
        embedding = DeepFace.represent(
            img_path=frame,
            model_name="Facenet",
            enforce_detection=False
        )[0]["embedding"]

        fdb.add_embedding(person_id, embedding, person_type)  # ✅ uses your new function
        return True

    except Exception as e:
        print("Enroll error:", e)
        return False


async def detect_faces(frame_bytes, filename, camera_id):
    np_arr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    try:
        faces = DeepFace.represent(
            img_path=frame,
            model_name="Facenet",
            enforce_detection=False
        )
    except Exception as e:
        print("Detect error:", e)
        return []

    all_embeddings = fdb.get_all_embeddings()
    detections = []

    for face_data in faces:
        incoming = np.array(face_data["embedding"])
        region = face_data.get("facial_area", {})

        best_id, best_type, best_score = None, "unknown", 0.0

        for person_id, emb_list in all_embeddings.items():
            for entry in emb_list:
                stored = np.array(entry["embedding"])
                score = float(np.dot(incoming, stored) / (np.linalg.norm(incoming) * np.linalg.norm(stored)))
                if score > best_score:
                    best_score = score
                    best_id = person_id
                    best_type = entry["type"]

        label = best_type if best_score > 0.6 else "unknown"
        person_id_out = best_id if best_score > 0.6 else None

        detections.append({
            "label":      label,
            "person_id":  person_id_out,
            "confidence": best_score,
            "bbox": [
                region.get("x", 0), region.get("y", 0),
                region.get("w", 0), region.get("h", 0)
            ]
        })

    return detections
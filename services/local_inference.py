# from ultralytics import YOLO
# from deepface import DeepFace
# import numpy as np
# import cv2
# from services import face_db

# # Load once (CPU)
# yolo_model = YOLO("yolov8n.pt")

# def cosine_similarity(a, b):
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# def detect_faces_local(frame, camera_id):
#     results = yolo_model(frame, classes=[0], conf=0.4, verbose=False)

#     detections = []

#     for r in results:
#         for box in r.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

#             face_crop = frame[y1:y2, x1:x2]

#             label = "unknown"
#             person_id = None
#             confidence = 0.0

#             try:
#                 if face_crop is not None and face_crop.size > 0:
#                     result = DeepFace.represent(
#                         img_path=face_crop,
#                         model_name="Facenet",
#                         enforce_detection=False
#                     )
#                     if not isinstance(result, list) or len(result) == 0:
#                         continue

#                     if not isinstance(result[0], dict) or "embedding" not in result[0]:
#                         print("❌ Invalid DeepFace output:", result)
#                         continue

#                     embedding = result[0]["embedding"]

#                     best_score = 0

#                     db = face_db.get_all_embeddings()

#                     if not isinstance(db, dict):
#                         print("❌ face_db corrupted:", db)
#                         return []

#                     for pid, data_list in db.items():
#                         for data in data_list:
#                             score = cosine_similarity(embedding, data["embedding"])

#                             if score > best_score:
#                                 best_score = score
#                                 person_id = pid
#                                 label = data["type"]

#                     if best_score > 0.6:
#                         confidence = float(best_score)

#             except Exception as e:
#                 print("DeepFace error:", e)

#             detections.append({
#                 "label": label,
#                 "person_id": person_id,
#                 "confidence": confidence,
#                 "bbox": [x1, y1, x2-x1, y2-y1],
#             })

#     return detections

from ultralytics import YOLO
from deepface import DeepFace
import numpy as np
import cv2
from services import face_db

# Load once (CPU)
yolo_model = YOLO("yolov8n.pt")

def cosine_similarity(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0
    return np.dot(a, b) / denom


def detect_faces_local(frame, camera_id):
    results = yolo_model(frame, classes=[0], conf=0.4, verbose=False)

    detections = []

    # ✅ FIX: move DB outside loops
    db = face_db.get_all_embeddings()

    if not isinstance(db, dict):
        print("❌ face_db corrupted:", db)
        return []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

            face_crop = frame[y1:y2, x1:x2]

            label = "unknown"
            person_id = None
            confidence = 0.0

            try:
                if face_crop is not None and face_crop.size > 0:
                    result = DeepFace.represent(
                        img_path=face_crop,
                        model_name="Facenet",
                        enforce_detection=False
                    )

                    # ✅ FIX: safe handling instead of direct continue
                    embedding = None

                    if isinstance(result, list) and len(result) > 0:
                        if isinstance(result[0], dict) and "embedding" in result[0]:
                            embedding = result[0]["embedding"]
                        else:
                            print("❌ Invalid DeepFace output:", result)
                    else:
                        print("❌ Empty DeepFace result")

                    if embedding is not None:
                        best_score = 0
                        person_id = None
                        # ✅ FIX: safe iteration
                        for pid, data_list in db.items():
                            if not isinstance(data_list, list):
                                continue

                            for data in data_list:
                                if not isinstance(data, dict):
                                    continue

                                score = cosine_similarity(embedding, data["embedding"])

                                if score > best_score:
                                    best_score = score
                                    person_id = pid
                                    # label = data["type"]

                        if best_score > 0.6:
                            label = "employee"
                            confidence = float(best_score)

                        else:
                            label = "visitor"
                            person_id = None    

            except Exception as e:
                print("DeepFace error:", e)

            detections.append({
                "label": label,
                "person_id": person_id,
                "confidence": confidence,
                "bbox": [x1, y1, x2-x1, y2-y1],
            })

    return detections
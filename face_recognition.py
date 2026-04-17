import csv
import os
from datetime import timedelta

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")

import cv2
import numpy as np
from deepface import DeepFace

DETECTOR_BACKEND = "retinaface"
PRIMARY_RECOGNITION_MODEL = os.environ.get("PRIMARY_FACE_MODEL", "Facenet512")
VERIFIER_RECOGNITION_MODEL = os.environ.get("VERIFIER_FACE_MODEL", "ArcFace")
RECOGNITION_MODELS = []
for model_name in (PRIMARY_RECOGNITION_MODEL, VERIFIER_RECOGNITION_MODEL):
    if model_name not in RECOGNITION_MODELS:
        RECOGNITION_MODELS.append(model_name)

MODEL_NORMALIZATION = {
    "ArcFace": "ArcFace",
    "Facenet512": "base",
    "SFace": "base",
}
CENTROID_WEIGHT = {
    "ArcFace": 0.35,
    "Facenet512": 0.35,
    "SFace": 0.35,
}


def normalize_embedding(embedding):
    embedding = np.array(embedding, dtype=np.float32)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding


def prepare_face_crop(face_image):
    face_crop = np.asarray(face_image)
    if face_crop.size == 0:
        return None

    if face_crop.ndim == 2:
        face_crop = cv2.cvtColor(face_crop, cv2.COLOR_GRAY2BGR)

    if face_crop.dtype != np.uint8:
        max_value = float(face_crop.max()) if face_crop.size else 0.0
        if max_value <= 1.0:
            face_crop = face_crop * 255.0
        face_crop = np.clip(face_crop, 0, 255).astype(np.uint8)

    return face_crop


def resize_frame_for_detection(image_bgr, max_side=850):
    if max_side is None or max_side <= 0:
        return image_bgr, 1.0

    height, width = image_bgr.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_side:
        return image_bgr, 1.0

    scale = max_side / float(longest_side)
    resized = cv2.resize(
        image_bgr,
        (int(width * scale), int(height * scale)),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def scale_bbox_to_original(bbox, scale):
    if scale == 1.0:
        return bbox

    x, y, w, h = bbox
    return (
        int(round(x / scale)),
        int(round(y / scale)),
        int(round(w / scale)),
        int(round(h / scale)),
    )


def extract_faces_from_image(
    image_bgr,
    min_face_size=(24, 24),
    expand_percentage=15,
    min_confidence=0.85,
):
    if image_bgr is None or image_bgr.size == 0:
        return []

    crops = []
    try:
        face_objs = DeepFace.extract_faces(
            img_path=image_bgr,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
            align=True,
            expand_percentage=expand_percentage,
            color_face="bgr",
            normalize_face=False,
        )
    except Exception as exc:
        print(f"Face extraction error: {exc}")
        return crops

    image_h, image_w = image_bgr.shape[:2]
    for face_obj in face_objs:
        confidence = float(face_obj.get("confidence", 1.0))
        if confidence < min_confidence:
            continue

        facial_area = face_obj.get("facial_area", {})
        x = int(facial_area.get("x", 0))
        y = int(facial_area.get("y", 0))
        w = int(facial_area.get("w", 0))
        h = int(facial_area.get("h", 0))
        if w < min_face_size[0] or h < min_face_size[1]:
            continue

        x = max(0, x)
        y = max(0, y)
        x_end = min(image_w, x + w)
        y_end = min(image_h, y + h)
        w = x_end - x
        h = y_end - y
        if w <= 0 or h <= 0:
            continue

        face_crop = prepare_face_crop(face_obj.get("face"))
        if face_crop is None:
            face_crop = image_bgr[y:y_end, x:x_end].copy()
            face_crop = prepare_face_crop(face_crop)

        if face_crop is None or face_crop.size == 0:
            continue

        crops.append((face_crop, (x, y, w, h)))

    return crops


def detect_faces_for_frame(frame, detection_max_side=850):
    attempts = []
    resized_frame, resized_scale = resize_frame_for_detection(frame, detection_max_side)
    attempts.append(
        {
            "frame": resized_frame,
            "scale": resized_scale,
            "expand_percentage": 15,
            "min_confidence": 0.85,
        }
    )

    if resized_scale != 1.0:
        attempts.append(
            {
                "frame": frame,
                "scale": 1.0,
                "expand_percentage": 15,
                "min_confidence": 0.80,
            }
        )

    attempts.append(
        {
            "frame": frame,
            "scale": 1.0,
            "expand_percentage": 20,
            "min_confidence": 0.75,
        }
    )

    for attempt in attempts:
        faces = extract_faces_from_image(
            attempt["frame"],
            expand_percentage=attempt["expand_percentage"],
            min_confidence=attempt["min_confidence"],
        )
        if not faces:
            continue

        scaled_faces = []
        for face_crop, bbox in faces:
            scaled_faces.append((face_crop, scale_bbox_to_original(bbox, attempt["scale"])))
        return scaled_faces

    return []


def get_embedding(face_crop_uint8, model_name):
    try:
        rep = DeepFace.represent(
            img_path=face_crop_uint8,
            model_name=model_name,
            detector_backend="skip",
            enforce_detection=False,
            normalization=MODEL_NORMALIZATION.get(model_name, "base"),
            l2_normalize=True,
        )
        if rep:
            return normalize_embedding(rep[0]["embedding"])
    except Exception as exc:
        print(f"Embedding error ({model_name}): {exc}")
    return None


def build_database(person_images_dict):
    print("Building database...")
    database = {
        "people": sorted(person_images_dict.keys()),
        "models": {model_name: {} for model_name in RECOGNITION_MODELS},
    }

    for person_id, images in person_images_dict.items():
        reference_faces = []
        for img_input in images:
            image_bgr = cv2.imread(img_input)
            if image_bgr is None:
                print(f"Warning: Could not read image {img_input}")
                continue

            crops = detect_faces_for_frame(image_bgr, detection_max_side=850)
            if not crops:
                print(f"Warning: No face detected in {img_input}")
                continue

            face_crop, _ = max(crops, key=lambda item: item[1][2] * item[1][3])
            reference_faces.append(face_crop)

        if not reference_faces:
            print(f"Warning: No valid faces found for '{person_id}'.")
            continue

        stored_count = 0
        for model_name in RECOGNITION_MODELS:
            embeddings = []
            for face_crop in reference_faces:
                emb = get_embedding(face_crop, model_name)
                if emb is not None:
                    embeddings.append(emb)

            if not embeddings:
                continue

            centroid = normalize_embedding(np.mean(np.stack(embeddings, axis=0), axis=0))
            database["models"][model_name][person_id] = {
                "embeddings": embeddings,
                "centroid": centroid,
            }
            stored_count = max(stored_count, len(embeddings))

        if stored_count > 0:
            print(f"Stored {stored_count} embeddings for '{person_id}'.")
        else:
            print(f"Warning: No embeddings were generated for '{person_id}'.")

    return database


def score_embedding(embedding, gallery, model_name):
    scores = {}
    centroid_weight = CENTROID_WEIGHT.get(model_name, 0.35)
    for person_id, person_data in gallery.items():
        embs = person_data["embeddings"]
        centroid = person_data["centroid"]
        max_sim = max(float(np.dot(embedding, db_emb)) for db_emb in embs)
        centroid_sim = float(np.dot(embedding, centroid))
        combined_score = ((1.0 - centroid_weight) * max_sim) + (centroid_weight * centroid_sim)
        scores[person_id] = combined_score
    return scores


def sort_scores(score_map):
    return sorted(score_map.items(), key=lambda item: item[1], reverse=True)


def identify_face(
    face_embeddings,
    database,
    threshold=0.48,
    similarity_gap=0.12,
    verifier_threshold=0.35,
):
    if not database["people"]:
        return "Unknown", 0.0, {}

    primary_embedding = face_embeddings.get(PRIMARY_RECOGNITION_MODEL)
    verifier_embedding = face_embeddings.get(VERIFIER_RECOGNITION_MODEL)
    if primary_embedding is None or verifier_embedding is None:
        return "Unknown", 0.0, {}

    primary_gallery = database["models"].get(PRIMARY_RECOGNITION_MODEL, {})
    verifier_gallery = database["models"].get(VERIFIER_RECOGNITION_MODEL, {})
    if not primary_gallery or not verifier_gallery:
        return "Unknown", 0.0, {}

    primary_scores = score_embedding(
        primary_embedding,
        primary_gallery,
        PRIMARY_RECOGNITION_MODEL,
    )
    verifier_scores = score_embedding(
        verifier_embedding,
        verifier_gallery,
        VERIFIER_RECOGNITION_MODEL,
    )
    if not primary_scores or not verifier_scores:
        return "Unknown", 0.0, {}

    primary_ranked = sort_scores(primary_scores)
    candidate_label, candidate_score = primary_ranked[0]
    second_primary_score = primary_ranked[1][1] if len(primary_ranked) > 1 else 0.0
    verifier_candidate_score = verifier_scores.get(candidate_label, 0.0)

    diagnostics = {
        "candidate_label": candidate_label,
        "primary_scores": primary_scores,
        "verifier_scores": verifier_scores,
        "primary_score": candidate_score,
        "primary_gap": candidate_score - second_primary_score,
        "verifier_candidate_score": verifier_candidate_score,
    }

    if candidate_score < threshold:
        return "Unknown", candidate_score, diagnostics

    if (candidate_score - second_primary_score) < similarity_gap:
        return "Unknown", candidate_score, diagnostics

    if verifier_candidate_score < verifier_threshold:
        return "Unknown", candidate_score, diagnostics

    return candidate_label, candidate_score, diagnostics


class FaceTracker:
    def __init__(
        self,
        iou_thresh=0.15,
        max_missing=60,
        appearance_thresh=0.35,
        carry_frames=2,
        switch_frames=2,
    ):
        self.tracks = []
        self.iou_thresh = iou_thresh
        self.max_missing = max_missing
        self.appearance_thresh = appearance_thresh
        self.carry_frames = carry_frames
        self.switch_frames = switch_frames
        self.next_track_id = 1

    def _compute_iou(self, box_a, box_b):
        x_a = max(box_a[0], box_b[0])
        y_a = max(box_a[1], box_b[1])
        x_b = min(box_a[0] + box_a[2], box_b[0] + box_b[2])
        y_b = min(box_a[1] + box_a[3], box_b[1] + box_b[3])
        inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
        box_a_area = box_a[2] * box_a[3]
        box_b_area = box_b[2] * box_b[3]
        return inter_area / float(box_a_area + box_b_area - inter_area + 1e-5)

    def _embedding_similarity(self, embedding_a, embedding_b):
        if embedding_a is None or embedding_b is None:
            return -1.0
        return float(np.dot(embedding_a, embedding_b))

    def _create_track(self, detection):
        track = {
            "track_id": self.next_track_id,
            "bbox": detection["bbox"],
            "appearance_embedding": detection["appearance_embedding"],
            "age": 0,
            "stable_label": detection["accepted_label"],
            "stable_score": detection["accepted_score"],
            "missed_recognitions": 0,
            "pending_label": None,
            "pending_hits": 0,
            "last_detection": detection,
        }
        self.next_track_id += 1
        return track

    def _update_label_state(self, track, detection):
        accepted_label = detection["accepted_label"]
        accepted_score = detection["accepted_score"]
        stable_label = track["stable_label"]

        if accepted_label == "Unknown":
            track["pending_label"] = None
            track["pending_hits"] = 0
            if stable_label != "Unknown":
                track["missed_recognitions"] += 1
                if track["missed_recognitions"] <= self.carry_frames:
                    return stable_label
            track["stable_label"] = "Unknown"
            track["stable_score"] = accepted_score
            return "Unknown"

        track["missed_recognitions"] = 0

        if stable_label in (None, "Unknown"):
            track["stable_label"] = accepted_label
            track["stable_score"] = accepted_score
            track["pending_label"] = None
            track["pending_hits"] = 0
            return accepted_label

        if accepted_label == stable_label:
            track["stable_score"] = max(track["stable_score"] * 0.9, accepted_score)
            track["pending_label"] = None
            track["pending_hits"] = 0
            return stable_label

        if accepted_score >= 0.75:
            track["stable_label"] = accepted_label
            track["stable_score"] = accepted_score
            track["pending_label"] = None
            track["pending_hits"] = 0
            return accepted_label

        if track["pending_label"] == accepted_label:
            track["pending_hits"] += 1
        else:
            track["pending_label"] = accepted_label
            track["pending_hits"] = 1

        if track["pending_hits"] >= self.switch_frames:
            track["stable_label"] = accepted_label
            track["stable_score"] = accepted_score
            track["pending_label"] = None
            track["pending_hits"] = 0

        return track["stable_label"]

    def _active_results(self):
        results = []
        for track in self.tracks:
            results.append(
                {
                    "track_id": track["track_id"],
                    "bbox": track["bbox"],
                    "label": track["stable_label"] or "Unknown",
                    "stable_score": track["stable_score"],
                }
            )
        return results

    def update(self, detections):
        new_tracks = []
        matched_track_indices = set()
        tracked_outputs = []

        for detection in detections:
            bbox = detection["bbox"]
            appearance_embedding = detection["appearance_embedding"]

            best_track_idx = -1
            best_match_score = -1.0
            for index, track in enumerate(self.tracks):
                if index in matched_track_indices:
                    continue
                iou = self._compute_iou(bbox, track["bbox"])
                if iou <= self.iou_thresh:
                    continue

                appearance_score = self._embedding_similarity(
                    appearance_embedding,
                    track["appearance_embedding"],
                )
                if appearance_score < self.appearance_thresh:
                    continue

                match_score = (0.65 * iou) + (0.35 * max(appearance_score, 0.0))
                if match_score > best_match_score:
                    best_match_score = match_score
                    best_track_idx = index

            if best_track_idx != -1:
                track = self.tracks[best_track_idx]
                matched_track_indices.add(best_track_idx)
            else:
                track = self._create_track(detection)

            track["bbox"] = bbox
            track["age"] = 0
            if appearance_embedding is not None:
                if track["appearance_embedding"] is None:
                    track["appearance_embedding"] = appearance_embedding
                else:
                    track["appearance_embedding"] = normalize_embedding(
                        (0.7 * track["appearance_embedding"]) + (0.3 * appearance_embedding)
                    )

            track["last_detection"] = detection
            final_label = self._update_label_state(track, detection)
            new_tracks.append(track)
            tracked_outputs.append(
                {
                    "track_id": track["track_id"],
                    "bbox": bbox,
                    "label": final_label,
                    "stable_score": track["stable_score"],
                    "candidate_label": detection["candidate_label"],
                    "accepted_label": detection["accepted_label"],
                    "accepted_score": detection["accepted_score"],
                    "primary_candidate_score": detection["primary_candidate_score"],
                    "primary_gap": detection["primary_gap"],
                    "verifier_candidate_score": detection["verifier_candidate_score"],
                }
            )

        for index, track in enumerate(self.tracks):
            if index in matched_track_indices:
                continue
            track["age"] += 1
            if track["age"] <= self.max_missing:
                new_tracks.append(track)

        self.tracks = new_tracks
        return tracked_outputs

    def step(self):
        active_tracks = []
        for track in self.tracks:
            track["age"] += 1
            if track["age"] <= self.max_missing:
                active_tracks.append(track)

        self.tracks = active_tracks
        return self._active_results()


def draw_predictions(frame, predictions):
    for prediction in predictions:
        x, y, w, h = prediction["bbox"]
        final_id = prediction["label"]
        color = (0, 255, 0) if final_id != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            frame,
            final_id,
            (x, max(10, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )


def seconds_to_timestamp(seconds):
    delta = timedelta(seconds=float(seconds))
    total_seconds = delta.total_seconds()
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def write_detection_csv(csv_path, records):
    fieldnames = [
        "frame_index",
        "timestamp_sec",
        "timestamp",
        "track_id",
        "stable_label",
        "stable_score",
        "accepted_label",
        "accepted_score",
        "candidate_label",
        "primary_candidate_score",
        "primary_gap",
        "verifier_candidate_score",
        "x",
        "y",
        "w",
        "h",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def build_event_summary(records, gap_tolerance_sec):
    events = []
    for record in records:
        label = record["stable_label"]
        if label == "Unknown":
            continue

        if (
            events
            and events[-1]["label"] == label
            and (record["timestamp_sec"] - events[-1]["end_time_sec"]) <= gap_tolerance_sec
        ):
            events[-1]["end_time_sec"] = record["timestamp_sec"]
            events[-1]["end_timestamp"] = record["timestamp"]
            events[-1]["frames"] += 1
            events[-1]["best_score"] = max(events[-1]["best_score"], record["stable_score"])
        else:
            events.append(
                {
                    "label": label,
                    "start_time_sec": record["timestamp_sec"],
                    "start_timestamp": record["timestamp"],
                    "end_time_sec": record["timestamp_sec"],
                    "end_timestamp": record["timestamp"],
                    "frames": 1,
                    "best_score": record["stable_score"],
                }
            )
    return events


def write_event_log(log_path, events):
    with open(log_path, "w", encoding="utf-8") as log_file:
        if not events:
            log_file.write("No recognized employees were found.\n")
            return

        for index, event in enumerate(events, start=1):
            log_file.write(
                f"{index}. {event['label']} from {event['start_timestamp']} to "
                f"{event['end_timestamp']} | analyzed_frames={event['frames']} | "
                f"best_stable_score={event['best_score']:.3f}\n"
            )


def process_video_pipeline(
    input_video,
    output_video,
    database,
    threshold=0.48,
    similarity_gap=0.12,
    analysis_fps=1.0,
    detection_max_side=850,
):
    print(f"Opening video: {input_video}")
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Failed to open video file.")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    frame_stride = max(1, int(round(fps / max(analysis_fps, 0.1))))
    tracker = FaceTracker(max_missing=frame_stride)
    frame_count = 0
    analyzed_frames = 0
    label_counter = {}
    analysis_records = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        should_analyze = frame_count == 1 or ((frame_count - 1) % frame_stride == 0)

        if should_analyze:
            analyzed_frames += 1
            detections = []
            detected_faces = detect_faces_for_frame(frame, detection_max_side=detection_max_side)
            for face_crop, bbox in detected_faces:
                face_embeddings = {}
                for model_name in RECOGNITION_MODELS:
                    emb = get_embedding(face_crop, model_name)
                    if emb is None:
                        face_embeddings = {}
                        break
                    face_embeddings[model_name] = emb

                if face_embeddings:
                    pred_id, score, diagnostics = identify_face(
                        face_embeddings,
                        database,
                        threshold,
                        similarity_gap,
                    )
                else:
                    pred_id, score, diagnostics = "Unknown", 0.0, {}

                detections.append(
                    {
                        "bbox": bbox,
                        "appearance_embedding": face_embeddings.get(PRIMARY_RECOGNITION_MODEL),
                        "accepted_label": pred_id,
                        "accepted_score": score,
                        "candidate_label": diagnostics.get("candidate_label", "Unknown"),
                        "primary_candidate_score": diagnostics.get("primary_score", 0.0),
                        "primary_gap": diagnostics.get("primary_gap", 0.0),
                        "verifier_candidate_score": diagnostics.get(
                            "verifier_candidate_score",
                            0.0,
                        ),
                    }
                )

            tracked_detections = tracker.update(detections)
            for tracked in tracked_detections:
                timestamp_sec = frame_count / fps
                x, y, w, h = tracked["bbox"]
                record = {
                    "frame_index": frame_count,
                    "timestamp_sec": round(timestamp_sec, 3),
                    "timestamp": seconds_to_timestamp(timestamp_sec),
                    "track_id": tracked["track_id"],
                    "stable_label": tracked["label"],
                    "stable_score": round(tracked["stable_score"], 4),
                    "accepted_label": tracked["accepted_label"],
                    "accepted_score": round(tracked["accepted_score"], 4),
                    "candidate_label": tracked["candidate_label"],
                    "primary_candidate_score": round(tracked["primary_candidate_score"], 4),
                    "primary_gap": round(tracked["primary_gap"], 4),
                    "verifier_candidate_score": round(tracked["verifier_candidate_score"], 4),
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                }
                analysis_records.append(record)
                label_counter[tracked["label"]] = label_counter.get(tracked["label"], 0) + 1

            predictions_to_draw = tracker._active_results()

            if analyzed_frames % 5 == 0:
                print(
                    f"Analyzed {analyzed_frames} frames "
                    f"({frame_count}/{total_frames or 'unknown'} total frames read)..."
                )
        else:
            predictions_to_draw = tracker.step()

        draw_predictions(frame, predictions_to_draw)
        out.write(frame)

    cap.release()
    out.release()

    csv_path = os.path.splitext(output_video)[0] + "_detections.csv"
    log_path = os.path.splitext(output_video)[0] + "_events.log"
    write_detection_csv(csv_path, analysis_records)
    analysis_interval_sec = frame_stride / fps if fps else 0.5
    events = build_event_summary(
        analysis_records,
        gap_tolerance_sec=max(analysis_interval_sec * 1.5, 0.5),
    )
    write_event_log(log_path, events)

    summary = {
        "input_video": input_video,
        "output_video": output_video,
        "fps": round(float(fps), 2),
        "frames_total": frame_count,
        "frames_analyzed": analyzed_frames,
        "analysis_fps": analysis_fps,
        "frame_stride": frame_stride,
        "label_counts": label_counter,
        "csv_path": csv_path,
        "log_path": log_path,
        "models": RECOGNITION_MODELS,
    }
    print(f"Video inference complete. Saved output to {output_video}")
    print(f"Summary: {summary}")
    return summary


if __name__ == "__main__":
    db_mapping = {}
    input_vid = "input.mp4"
    output_vid = "output.mp4"

    # Uncomment to run a local demo.
    # face_db = build_database(db_mapping)
    # process_video_pipeline(input_vid, output_vid, face_db)

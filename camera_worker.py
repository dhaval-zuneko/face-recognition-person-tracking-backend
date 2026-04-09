import cv2
import os
from services.local_inference import detect_faces_local

def run_camera(camera_id, source):
    cap = cv2.VideoCapture(source)

    print(f"🚀 Starting camera: {camera_id}")

    # ✅ Create output folder
    os.makedirs("processed_videos", exist_ok=True)

    # Get video properties
    ret, frame = cap.read()
    if not ret:
        print(f"❌ Cannot read video {source}")
        return

    h, w, _ = frame.shape

    output_path = f"processed_videos/{camera_id}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 10, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"❌ Camera {camera_id} ended")
            break

        if frame is None:
            continue

        detections = detect_faces_local(frame, camera_id)

        # ✅ DRAW BOXES
        for det in detections:
            x, y, w_box, h_box = det.get("bbox", [0, 0, 0, 0])
            label = det.get("label", "unknown")

            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # ✅ SAVE FRAME
        out.write(frame)

        print(f"[{camera_id}] → {len(detections)} detections")

    cap.release()
    out.release()

    print(f"✅ Saved video: {output_path}")
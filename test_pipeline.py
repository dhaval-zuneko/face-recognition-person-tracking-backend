import argparse
from pathlib import Path

from face_recognition import build_database, process_video_pipeline

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def collect_employee_images(base_dir):
    db_mapping = {}
    for emp_dir in sorted(base_dir.iterdir()):
        if not emp_dir.is_dir() or not emp_dir.name.lower().startswith("emp_"):
            continue

        images = [
            str(path)
            for path in sorted(emp_dir.iterdir())
            if path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if images:
            db_mapping[emp_dir.name] = images
            print(f"Found {len(images)} images for {emp_dir.name}")

    return db_mapping


def parse_args():
    parser = argparse.ArgumentParser(description="CPU ArcFace + RetinaFace video test")
    parser.add_argument("--threshold", type=float, default=0.48)
    parser.add_argument("--similarity-gap", type=float, default=0.12)
    parser.add_argument("--analysis-fps", type=float, default=1.0)
    parser.add_argument("--detection-max-side", type=int, default=850)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for annotated output videos. Defaults to ./outputs",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    output_dir = args.output_dir or (base_dir / "outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    db_mapping = collect_employee_images(base_dir)
    if not db_mapping:
        raise SystemExit("No employee image folders were found in the project directory.")

    print("Building face database...")
    face_db = build_database(db_mapping)
    if not face_db:
        raise SystemExit("Face database build failed. Check the employee reference images.")

    videos_to_test = sorted(base_dir.glob("*.mp4"))
    if not videos_to_test:
        raise SystemExit("No .mp4 files found in the project directory.")

    for video_path in videos_to_test:
        output_vid = output_dir / f"output_{video_path.name}"
        print(f"\n--- Processing {video_path.name} ---")
        summary = process_video_pipeline(
            str(video_path),
            str(output_vid),
            face_db,
            threshold=args.threshold,
            similarity_gap=args.similarity_gap,
            analysis_fps=args.analysis_fps,
            detection_max_side=args.detection_max_side,
        )
        if summary is not None:
            print(f"Label counts for {video_path.name}: {summary['label_counts']}")


if __name__ == "__main__":
    main()

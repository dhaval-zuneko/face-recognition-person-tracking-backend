from multiprocessing import Process
from camera_worker import run_camera

CAMERAS = [
    ("entry", "video1.mp4"),
    ("floor1", "video2.mp4"),
    ("floor2", "video3.mp4"),
]

def start_cameras():
    processes = []

    for cam_id, src in CAMERAS:
        p = Process(target=run_camera, args=(cam_id, src))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    start_cameras()
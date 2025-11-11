from __future__ import annotations

import csv
import time
from pathlib import Path

import cv2

from aircursor.core.hand_tracker import HandTracker, HandTrackerConfig
from aircursor.utils.landmark_preprocess import (
    LandmarkPacket,
    normalize_landmarks,
    vectorize_landmarks,
)


def save_landmark(packet: LandmarkPacket, target_dir: Path, label: str) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    feature = vectorize_landmarks(normalize_landmarks(packet.landmarks))
    timestamp = int(time.time() * 1000)
    csv_path = target_dir / f"{label}_{timestamp}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z"])
        for (x, y, z) in feature.reshape(-1, 3):
            writer.writerow([float(x), float(y), float(z)])


def main() -> None:
    label = input("请输入要采集的静态手势标签（如 open/closed/...）：").strip()
    output_dir = Path("data/static") / label

    tracker = HandTracker(HandTrackerConfig())
    cap = cv2.VideoCapture(0)
    print("按空格记录一帧，按 q 退出。")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            packet = tracker.process(frame)
            if packet:
                cv2.putText(frame, "Detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Collect Static Gesture", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                if packet:
                    save_landmark(packet, output_dir, label)
                    print("已保存关键点。")
            elif key == ord("q"):
                break
    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from aircursor.core.hand_tracker import HandTracker, HandTrackerConfig
from aircursor.utils.landmark_preprocess import compute_hand_center
from aircursor.utils.trajectory_recorder import TrajectoryRecorder


def main() -> None:
    label = input("请输入动态手势标签（如 swipe_up/...）：").strip()
    output_dir = Path("data/dynamic") / label
    output_dir.mkdir(parents=True, exist_ok=True)

    tracker = HandTracker(HandTrackerConfig())
    recorder = TrajectoryRecorder(maxlen=96)
    cap = cv2.VideoCapture(0)
    print("按 r 开始/停止录制，按 q 退出。")
    recording = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            packet = tracker.process(frame)
            if packet and recording:
                cx, cy = compute_hand_center(packet.landmarks)
                recorder.append(cx, cy, time.time())
                cv2.putText(frame, "Recording...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Collect Dynamic Gesture", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                if recording and recorder.to_numpy().size:
                    data = recorder.to_numpy()
                    ts = int(time.time() * 1000)
                    file_path = output_dir / f"{label}_{ts}.npy"
                    np.save(file_path, data)
                    meta_path = output_dir / f"{label}_{ts}.json"
                    meta_path.write_text(json.dumps({"label": label, "length": data.shape[0]}))
                    print(f"已保存 {file_path}")
                    recorder.clear()
                    recording = False
                else:
                    recorder.clear()
                    recording = True
            elif key == ord("q"):
                break

    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


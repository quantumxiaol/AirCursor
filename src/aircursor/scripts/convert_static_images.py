from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import mediapipe as mp
import numpy as np

from aircursor.utils.landmark_preprocess import mp_landmarks_to_np, normalize_landmarks

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class ConvertConfig:
    input_root: Path
    output_format: str = "csv"  # csv 或 npy
    overwrite: bool = False
    debug: bool = False


def iter_image_files(root: Path) -> Iterable[Path]:
    for label_dir in sorted(root.iterdir()):
        if not label_dir.is_dir():
            continue
        for file_path in sorted(label_dir.iterdir()):
            if file_path.suffix.lower() in ALLOWED_EXTS:
                yield file_path


def save_csv(feature: np.ndarray, out_path: Path, overwrite: bool) -> None:
    if out_path.exists() and not overwrite:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = "x,y,z"
    data = feature.reshape(-1, 3)
    np.savetxt(out_path, data, delimiter=",", header=header, comments="", fmt="%.6f")


def save_npy(feature: np.ndarray, out_path: Path, overwrite: bool) -> None:
    if out_path.exists() and not overwrite:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, feature.reshape(-1, 3))


def process_image(
    image_path: Path,
    detector: mp.solutions.hands.Hands,
    cfg: ConvertConfig,
) -> Optional[Path]:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        if cfg.debug:
            print(f"[跳过] 无法读取图像 {image_path}")
        return None

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = detector.process(image_rgb)
    if not results.multi_hand_landmarks:
        if cfg.debug:
            print(f"[跳过] 未检测到手势 {image_path}")
        return None

    landmarks = mp_landmarks_to_np(results.multi_hand_landmarks[0])
    norm = normalize_landmarks(landmarks)
    feature = norm.astype(np.float32)

    if cfg.output_format == "csv":
        out_path = image_path.with_suffix(".csv")
        save_csv(feature, out_path, cfg.overwrite)
    else:
        out_path = image_path.with_suffix(".npy")
        save_npy(feature, out_path, cfg.overwrite)
    return out_path


def convert_dataset(cfg: ConvertConfig) -> None:
    if not cfg.input_root.exists():
        raise FileNotFoundError(f"输入目录 {cfg.input_root} 不存在。")

    mp_hands = mp.solutions.hands
    converted = 0
    skipped = 0

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as detector:
        for image_path in iter_image_files(cfg.input_root):
            result = process_image(image_path, detector, cfg)
            if result is None:
                skipped += 1
            else:
                converted += 1
                if cfg.debug:
                    print(f"[保存] {image_path.name} -> {result.name}")

    print(
        f"转换完成：成功 {converted} 张，跳过 {skipped} 张。"
        f" 输出格式: {cfg.output_format.upper()}"
    )


def parse_args(argv: list[str]) -> ConvertConfig:
    parser = argparse.ArgumentParser(description="将静态手势图片转换为关键点特征")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/static"),
        help="静态手势图像根目录（默认 data/static）",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "npy"],
        default="csv",
        help="输出格式（csv 或 npy，默认 csv）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若目标文件已存在是否覆盖",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="打印调试信息",
    )
    args = parser.parse_args(argv)
    return ConvertConfig(
        input_root=args.input_root,
        output_format=args.format,
        overwrite=args.overwrite,
        debug=args.debug,
    )


def main(argv: Optional[list[str]] = None) -> None:
    cfg = parse_args(sys.argv[1:] if argv is None else argv)
    convert_dataset(cfg)


if __name__ == "__main__":
    main()


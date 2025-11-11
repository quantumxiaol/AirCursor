from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from aircursor.utils.landmark_preprocess import normalize_landmarks


DEFAULT_GESTURE_MAP = {
    "open": ["palm", "stop", "stop_inverted"],
    "closed": ["fist", "rock", "grabbing", "grip"],
    "peace": ["peace", "peace_inverted", "two_up", "two_up_inverted"],
    "three": ["three", "three2", "three3"],
}

ALLOWED_STATIC_GESTURES = set(DEFAULT_GESTURE_MAP.keys())


@dataclass
class ImportConfig:
    dataset_root: Path
    annotations: Path
    gesture_map: Dict[str, List[str]]
    output_root: Path
    copy_images: bool
    overwrite: bool
    limit_per_class: Optional[int]
    split: Optional[str]


def parse_gesture_map(raw: Optional[Sequence[str]]) -> Dict[str, List[str]]:
    if not raw:
        return DEFAULT_GESTURE_MAP.copy()
    result: Dict[str, List[str]] = {}
    for item in raw:
        if "=" not in item:
            raise ValueError(f"手势映射格式错误：{item}，应为 target=src1,src2")
        target, sources = item.split("=", 1)
        target = target.strip()
        if target not in ALLOWED_STATIC_GESTURES:
            raise ValueError(f"不支持的目标手势：{target}，可选 {sorted(ALLOWED_STATIC_GESTURES)}")
        source_labels = [s.strip() for s in sources.split(",") if s.strip()]
        if not source_labels:
            raise ValueError(f"{item} 未提供源手势列表")
        result[target] = source_labels
    for key in ALLOWED_STATIC_GESTURES:
        result.setdefault(key, DEFAULT_GESTURE_MAP[key])
    return result


def ensure_output_dir(output_root: Path, label: str) -> Path:
    target_dir = output_root / "static" / label
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def load_annotations(path: Path) -> Dict[str, dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("注释文件格式不正确：根节点应为字典。")
    return data


def convert_landmarks(landmarks: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.array(landmarks, dtype=np.float32)
    if arr.shape != (21, 3):
        raise ValueError(f"关键点形状异常：{arr.shape}, 期望 (21, 3)")
    normalized = normalize_landmarks(arr)
    return normalized


def write_feature(feature: np.ndarray, out_path: Path, overwrite: bool) -> None:
    if out_path.exists() and not overwrite:
        return
    header = "x,y,z"
    np.savetxt(out_path, feature, delimiter=",", header=header, comments="", fmt="%.6f")


def copy_image(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def import_from_hagrid(cfg: ImportConfig) -> None:
    annotations = load_annotations(cfg.annotations)
    label_to_target: Dict[str, str] = {}
    for target, sources in cfg.gesture_map.items():
        for src in sources:
            label_to_target[src] = target

    counters = {target: 0 for target in cfg.gesture_map}
    skipped_no_landmarks = 0
    skipped_unknown = 0
    processed = 0

    for image_name, record in annotations.items():
        labels = record.get("labels", [])
        landmarks_all = record.get("hand_landmarks")
        if not landmarks_all:
            skipped_no_landmarks += 1
            continue

        gesture_pairs: List[Tuple[str, Sequence[Sequence[float]]]] = []
        for label, hand_landmarks in zip(labels, landmarks_all):
            target = label_to_target.get(label)
            if target is None:
                skipped_unknown += 1
                continue
            gesture_pairs.append((target, hand_landmarks))

        if not gesture_pairs:
            continue

        for target, hand_landmarks in gesture_pairs:

            limit = cfg.limit_per_class
            if limit is not None and counters[target] >= limit:
                continue

            try:
                normalized = convert_landmarks(hand_landmarks)
            except Exception as exc:
                print(f"[警告] 跳过 {image_name}: {exc}")
                continue

            target_dir = ensure_output_dir(cfg.output_root, target)

            filename_stem = f"hagrid_{target}_{counters[target]:06d}"
            csv_path = target_dir / f"{filename_stem}.csv"
            write_feature(normalized, csv_path, cfg.overwrite)

            if cfg.copy_images:
                label_sources = cfg.gesture_map[target]
                matched_source = next((src for src in label_sources if src in labels), None)
                if matched_source:
                    relative_image = Path(matched_source) / f"{image_name}.jpg"
                else:
                    relative_image = Path(labels[0]) / f"{image_name}.jpg"
                image_path = cfg.dataset_root / relative_image
                if image_path.exists():
                    dst = target_dir / f"{filename_stem}.jpg"
                    copy_image(image_path, dst, cfg.overwrite)
                else:
                    print(f"[警告] 未找到原图 {image_path}")

            counters[target] += 1
            processed += 1

    summary_lines = [
        "导入完成：",
        *(f"  {gesture}: {count} 条" for gesture, count in counters.items()),
        f"跳过（无关键点）: {skipped_no_landmarks}",
        f"跳过（未映射标签）: {skipped_unknown}",
        f"总计写入: {processed}",
    ]
    print("\n".join(summary_lines))


def parse_args() -> ImportConfig:
    parser = argparse.ArgumentParser(description="从 HaGRID 注释导入静态手势特征")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="HaGRID 数据集根目录（包含各手势子目录）",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="HaGRID 注释 JSON 文件（train/val/test 中之一）",
    )
    parser.add_argument(
        "--gesture-map",
        nargs="*",
        help="手势映射，例如 open=palm,stop closed=fist,rock",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data"),
        help="输出根目录（默认 data）",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="是否同时复制原图（用于可视化），默认不复制",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="是否覆盖已有 CSV/图像文件，默认不覆盖",
    )
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=None,
        help="每个目标手势最多导入多少条（默认不限）",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default=None,
        help="仅用于标识当前注释来源，可忽略",
    )

    args = parser.parse_args()
    gesture_map = parse_gesture_map(args.gesture_map)

    return ImportConfig(
        dataset_root=args.dataset_root,
        annotations=args.annotations,
        gesture_map=gesture_map,
        output_root=args.output_root,
        copy_images=args.copy_images,
        overwrite=args.overwrite,
        limit_per_class=args.limit_per_class,
        split=args.split,
    )


def main() -> None:
    cfg = parse_args()
    import_from_hagrid(cfg)


if __name__ == "__main__":
    main()


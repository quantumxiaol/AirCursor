#!/usr/bin/env python3
"""AirCursor 模型自动下载脚本

自动下载所需的模型文件到 weights/ 目录。
"""

import argparse
import sys
from pathlib import Path
from urllib.request import urlretrieve

# 模型下载配置
MODELS = {
    "hand_landmarker.task": {
        "url": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        "size": "~26MB",
        "description": "MediaPipe 手部关键点检测模型",
    },
    "hand_detector.onnx": {
        "url": "https://raw.githubusercontent.com/ai-forever/dynamic_gestures/main/models/hand_detector.onnx",
        "size": "~9MB",
        "description": "动态手势 - 手部检测模型（ONNX）",
    },
    "crops_classifier.onnx": {
        "url": "https://raw.githubusercontent.com/ai-forever/dynamic_gestures/main/models/crops_classifier.onnx",
        "size": "~1.5MB",
        "description": "动态手势 - 手势分类模型（ONNX）",
    },
    "ResNet18.pth": {
        "url": "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/models/ResNet18.pth",
        "size": "~43MB",
        "description": "HaGRID ResNet18 静态手势分类模型",
    },
}


def download_progress(block_num, block_size, total_size):
    """显示下载进度"""
    downloaded = block_num * block_size
    percent = min(downloaded * 100 / total_size, 100) if total_size > 0 else 0
    bar_length = 50
    filled = int(bar_length * percent / 100)
    bar = "█" * filled + "░" * (bar_length - filled)
    
    size_mb = downloaded / (1024 * 1024)
    total_mb = total_size / (1024 * 1024)
    
    print(f"\r  [{bar}] {percent:.1f}% ({size_mb:.1f}/{total_mb:.1f} MB)", end="", flush=True)


def download_model(name: str, config: dict, weights_dir: Path, force: bool = False) -> bool:
    """下载单个模型文件
    
    Args:
        name: 模型文件名
        config: 模型配置信息
        weights_dir: 权重目录
        force: 是否强制重新下载
        
    Returns:
        bool: 是否成功下载
    """
    file_path = weights_dir / name
    
    # 检查文件是否已存在
    if file_path.exists() and not force:
        print(f"✓ {name} 已存在，跳过下载")
        return True
    
    print(f"\n{'=' * 70}")
    print(f"📦 下载: {name}")
    print(f"📝 说明: {config['description']}")
    print(f"📊 大小: {config['size']}")
    print(f"🔗 来源: {config['url']}")
    print(f"{'=' * 70}")
    
    try:
        # 创建临时文件
        temp_path = file_path.with_suffix(file_path.suffix + ".tmp")
        
        # 下载文件
        print(f"⏬ 开始下载...")
        urlretrieve(config["url"], temp_path, reporthook=download_progress)
        print()  # 换行
        
        # 重命名为最终文件名
        temp_path.rename(file_path)
        
        print(f"✅ 下载完成: {name}")
        return True
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  下载被中断: {name}")
        # 清理临时文件
        if temp_path.exists():
            temp_path.unlink()
        return False
        
    except Exception as e:
        print(f"\n\n❌ 下载失败: {name}")
        print(f"   错误: {e}")
        # 清理临时文件
        if temp_path.exists():
            temp_path.unlink()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="AirCursor 模型自动下载工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 下载所有模型
  python download_models.py
  
  # 只下载特定模型
  python download_models.py --models hand_landmarker.task ResNet18.pth
  
  # 强制重新下载所有模型
  python download_models.py --force
  
  # 下载到自定义目录
  python download_models.py --output-dir /path/to/weights
        """
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()),
        help="指定要下载的模型（默认下载全部）",
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("weights"),
        help="模型保存目录（默认: weights/）",
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新下载已存在的文件",
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出所有可用的模型",
    )
    
    args = parser.parse_args()
    
    # 列出模型信息
    if args.list:
        print("\n" + "=" * 70)
        print("📋 可用模型列表")
        print("=" * 70)
        for name, config in MODELS.items():
            print(f"\n📦 {name}")
            print(f"   描述: {config['description']}")
            print(f"   大小: {config['size']}")
            print(f"   URL:  {config['url']}")
        print("\n" + "=" * 70)
        return 0
    
    # 创建权重目录
    weights_dir = args.output_dir
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("🚀 AirCursor 模型下载工具")
    print("=" * 70)
    print(f"📁 目标目录: {weights_dir.absolute()}")
    
    # 确定要下载的模型
    models_to_download = args.models if args.models else list(MODELS.keys())
    
    print(f"📦 待下载模型数: {len(models_to_download)}")
    if args.force:
        print("⚠️  强制重新下载模式")
    print()
    
    # 下载模型
    success_count = 0
    failed_models = []
    
    for model_name in models_to_download:
        config = MODELS[model_name]
        if download_model(model_name, config, weights_dir, args.force):
            success_count += 1
        else:
            failed_models.append(model_name)
    
    # 显示总结
    print("\n" + "=" * 70)
    print("📊 下载总结")
    print("=" * 70)
    print(f"✅ 成功: {success_count}/{len(models_to_download)}")
    
    if failed_models:
        print(f"❌ 失败: {len(failed_models)}")
        print(f"   失败的模型: {', '.join(failed_models)}")
        print("\n💡 提示: 可以重新运行脚本继续下载失败的模型")
        return 1
    else:
        print("\n🎉 所有模型下载完成！")
        print(f"📂 模型位置: {weights_dir.absolute()}")
        print("\n🚀 现在可以运行 AirCursor 了：")
        print("   python -m aircursor")
        print("   python -m aircursor.scripts.preview_static_gestures --mirror")
        print("   python -m aircursor.scripts.preview_dynamic_gestures --mirror \\")
        print("     --detector weights/hand_detector.onnx \\")
        print("     --classifier weights/crops_classifier.onnx")
        return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  程序被用户中断")
        sys.exit(130)


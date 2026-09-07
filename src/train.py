"""YOLOv8s 在 GC10-DET 上的训练脚本。

超参数取自课程设计报告表 6，和实际跑出 docs/figures/training-curves.png 的那一次
训练配置一致。想复现就别动这些数，想调参再动。

用法：
    python src/train.py
    python src/train.py --data data/gc10det.yaml --epochs 100
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser(description="训练 YOLOv8s 检测金属表面缺陷")
    p.add_argument("--data", default=str(ROOT / "data" / "gc10det.yaml"),
                   help="数据集配置 yaml")
    p.add_argument("--weights", default="yolov8s.pt",
                   help="预训练权重。s 版约 11M 参数，兼顾 8GB 显存和过拟合风险")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=1024,
                   help="原图约 2048x2048，缺陷偏小，缩到 1024 保细节")
    p.add_argument("--batch", type=int, default=4,
                   help="1024 输入下 8GB 显存的上限")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--close-mosaic", type=int, default=10,
                   help="最后 N 轮关闭 Mosaic，让模型收敛到真实分布")
    p.add_argument("--amp", action="store_true",
                   help="开混合精度。报告里那次训练是关掉的，为了稳定")
    p.add_argument("--device", default="0")
    p.add_argument("--project", default=str(ROOT / "runs" / "detect"))
    p.add_argument("--name", default="train")
    return p.parse_args()


def main():
    args = parse_args()

    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        optimizer="auto",       # 实际解析为 SGD + momentum
        lr0=args.lr0,
        amp=args.amp,
        close_mosaic=args.close_mosaic,
        device=args.device,
        project=args.project,
        name=args.name,
        plots=True,             # 训练结束自动出 results.png / 混淆矩阵 / PR 曲线
    )

    print("训练完成。best.pt 在 runs/detect/<name>/weights/ 下，按验证集 mAP 最高那轮保存。")


if __name__ == "__main__":
    main()

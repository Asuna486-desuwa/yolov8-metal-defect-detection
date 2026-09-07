"""用训练好的权重对图片做推理并保存可视化结果。

用法：
    python src/predict.py --weights runs/detect/train/weights/best.pt --source path/to/img.jpg
    python src/predict.py --weights best.pt --source path/to/dir/ --conf 0.25
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser(description="金属表面缺陷检测推理")
    p.add_argument("--weights", required=True)
    p.add_argument("--source", required=True, help="单张图片、目录或视频")
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--device", default="0")
    p.add_argument("--project", default=str(ROOT / "runs" / "predict"))
    p.add_argument("--name", default="exp")
    return p.parse_args()


def main():
    args = parse_args()

    model = YOLO(args.weights)
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=True,
        project=args.project,
        name=args.name,
    )

    total = 0
    for r in results:
        n = len(r.boxes)
        total += n
        if n:
            names = [model.names[int(c)] for c in r.boxes.cls]
            confs = [f"{float(c):.2f}" for c in r.boxes.conf]
            pairs = ", ".join(f"{a} {b}" for a, b in zip(names, confs))
            print(f"{Path(r.path).name}: {n} 处缺陷 [{pairs}]")
        else:
            print(f"{Path(r.path).name}: 未检出")

    print(f"\n共 {len(results)} 张图，检出 {total} 处缺陷。带框结果已存到 {args.project}/{args.name}/")


if __name__ == "__main__":
    main()

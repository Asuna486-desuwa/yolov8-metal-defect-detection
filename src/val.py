"""在测试集上评估已训练的权重。

跑完会打印整体的 P / R / mAP@0.5 / mAP@0.5:0.95，以及逐类别指标，
对应报告里的表 7 和表 8。同时在输出目录下生成 PR 曲线和归一化混淆矩阵。

用法：
    python src/val.py --weights runs/detect/train/weights/best.pt
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

CLASS_NAMES_CN = [
    "冲孔缺陷", "焊缝缺陷", "月牙弯", "水斑", "油斑",
    "丝斑", "异物", "压痕", "折痕", "咬折",
]


def parse_args():
    p = argparse.ArgumentParser(description="评估 YOLOv8 缺陷检测模型")
    p.add_argument("--weights", required=True, help="训练得到的 best.pt")
    p.add_argument("--data", default=str(ROOT / "data" / "gc10det.yaml"))
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", default="0")
    return p.parse_args()


def main():
    args = parse_args()

    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        plots=True,
    )

    box = metrics.box
    print(f"\n{args.split} 集整体指标")
    print(f"  Precision      {box.mp:.3f}")
    print(f"  Recall         {box.mr:.3f}")
    print(f"  mAP@0.5        {box.map50:.3f}")
    print(f"  mAP@0.5:0.95   {box.map:.3f}")

    print("\n逐类别指标")
    print(f"{'类别':<16}{'中文':<10}{'P':>8}{'R':>8}{'mAP@0.5':>10}")
    for i, cls_id in enumerate(box.ap_class_index):
        p, r, ap50 = box.p[i], box.r[i], box.ap50[i]
        en = model.names[int(cls_id)]
        cn = CLASS_NAMES_CN[int(cls_id)]
        print(f"{en:<16}{cn:<10}{p:>8.3f}{r:>8.3f}{ap50:>10.3f}")


if __name__ == "__main__":
    main()

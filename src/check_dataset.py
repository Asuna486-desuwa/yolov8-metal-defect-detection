"""训练前的数据集自检。

踩过的坑：images 和 labels 文件名对不上、标签坐标没归一化、data.yaml 里的 nc
和 names 数量不一致，这几种问题训练时不会立刻报错，只会让指标莫名其妙地差。
所以先跑一遍这个脚本。

用法：
    python src/check_dataset.py
    python src/check_dataset.py --data data/gc10det.yaml
"""

import argparse
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args():
    p = argparse.ArgumentParser(description="检查 YOLO 数据集完整性")
    p.add_argument("--data", default=str(ROOT / "data" / "gc10det.yaml"))
    return p.parse_args()


def check_split(images_dir, labels_dir, nc, problems):
    """核对一个划分：图片标签是否成对，标签内容是否合法。"""
    if not images_dir.is_dir():
        problems.append(f"缺图片目录 {images_dir}")
        return None
    if not labels_dir.is_dir():
        problems.append(f"缺标签目录 {labels_dir}")
        return None

    imgs = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    stems = {p.stem for p in imgs}
    label_stems = {p.stem for p in labels_dir.glob("*.txt")}

    for s in sorted(stems - label_stems):
        problems.append(f"{images_dir.name}: 图片 {s} 没有对应标签")
    for s in sorted(label_stems - stems):
        problems.append(f"{labels_dir.name}: 标签 {s}.txt 没有对应图片")

    counter = Counter()
    empty = 0
    for lp in sorted(labels_dir.glob("*.txt")):
        lines = [ln for ln in lp.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            empty += 1
            continue
        for i, ln in enumerate(lines, 1):
            parts = ln.split()
            if len(parts) != 5:
                problems.append(f"{lp.name}:{i} 应为 5 个数，实际 {len(parts)} 个")
                continue
            try:
                cls = int(float(parts[0]))
                vals = [float(v) for v in parts[1:]]
            except ValueError:
                problems.append(f"{lp.name}:{i} 有非数值字段")
                continue
            if not 0 <= cls < nc:
                problems.append(f"{lp.name}:{i} 类别编号 {cls} 超出 0~{nc - 1}")
            if any(not 0.0 <= v <= 1.0 for v in vals):
                problems.append(f"{lp.name}:{i} 坐标未归一化到 [0,1]：{vals}")
            counter[cls] += 1

    return {"images": len(imgs), "instances": sum(counter.values()),
            "per_class": counter, "empty_labels": empty}


def main():
    args = parse_args()
    with open(args.data, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg["path"])
    nc = int(cfg["nc"])
    names = cfg["names"]
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]

    problems = []
    if len(names) != nc:
        problems.append(f"nc={nc} 但 names 有 {len(names)} 项，两者必须一致")
    if not root.is_dir():
        problems.append(f"数据集根目录不存在：{root}，先改 data yaml 里的 path")

    stats, total_imgs = {}, 0
    for key, label in (("train", "train"), ("val", "valid"), ("test", "test")):
        rel = cfg.get(key)
        if rel is None:
            continue
        images_dir = root / rel
        labels_dir = Path(str(images_dir).replace("images", "labels"))
        s = check_split(images_dir, labels_dir, nc, problems)
        if s:
            stats[label] = s
            total_imgs += s["images"]

    for label, s in stats.items():
        share = f"{s['images'] / total_imgs * 100:.1f}%" if total_imgs else "-"
        print(f"{label:<6} {s['images']:>5} 张 ({share:>6})  "
              f"{s['instances']:>5} 个标注框  空标签 {s['empty_labels']}")
    if total_imgs:
        print(f"{'合计':<6} {total_imgs:>5} 张")

    if stats:
        print("\n各类别实例数")
        for cls in range(nc):
            cells = "  ".join(
                f"{label} {s['per_class'].get(cls, 0):>4}" for label, s in stats.items())
            print(f"  {cls} {names[cls]:<16}{cells}")

    if problems:
        print(f"\n发现 {len(problems)} 个问题：")
        for p in problems[:50]:
            print(f"  - {p}")
        if len(problems) > 50:
            print(f"  ... 另有 {len(problems) - 50} 个未列出")
        raise SystemExit(1)

    print("\n数据集检查通过，可以开始训练。")


if __name__ == "__main__":
    main()

"""t-SNE 可视化 YOLOv8 学到的深层特征。

思路：检测指标只告诉你模型准不准，看不出它把哪几类混在一起了。所以直接把检测头
前一层的特征抓出来降到二维看聚类，能不能分开一目了然。

流程：
  1. 加载 best.pt，在 model.model[-2]（检测头前一层）挂 PyTorch forward hook
  2. 按 YOLO 标签把每个缺陷区域从原图裁出来，逐个送进模型前向
  3. hook 收到的特征图做全局平均池化，得到每个缺陷一条特征向量
  4. StandardScaler 标准化 -> PCA 降到 50 维去噪 -> t-SNE 降到 2 维
  5. 按类别上色画散点

perplexity 按样本量自适应，样本足够时取 30。

用法：
    python src/tsne_features.py --weights runs/detect/train/weights/best.pt --split test
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# 裁剪框太小的样本特征没有意义，直接丢掉
MIN_CROP_PX = 8


def parse_args():
    p = argparse.ArgumentParser(description="t-SNE 可视化 YOLOv8 深层特征")
    p.add_argument("--weights", required=True)
    p.add_argument("--data", default=str(ROOT / "data" / "gc10det.yaml"))
    p.add_argument("--split", default="test", choices=["train", "valid", "test"])
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--pca-dims", type=int, default=50)
    p.add_argument("--max-samples", type=int, default=0,
                   help="最多取多少个缺陷区域，0 表示全部")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(ROOT / "results" / "tsne-features.png"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_dataset_cfg(data_yaml, split):
    """返回 (images_dir, labels_dir, class_names)。"""
    with open(data_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg["path"])
    key = "val" if split == "valid" else split
    rel = cfg.get(key)
    if rel is None:
        raise KeyError(f"{data_yaml} 里没有 {key} 这一项")

    images_dir = root / rel
    labels_dir = Path(str(images_dir).replace("images", "labels"))
    if not images_dir.is_dir():
        raise FileNotFoundError(f"找不到图片目录 {images_dir}，先改 data yaml 里的 path")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"找不到标签目录 {labels_dir}")

    names = cfg["names"]
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]
    return images_dir, labels_dir, names


def read_labels(label_path):
    """读 YOLO 标签：每行 class cx cy w h，坐标已归一化到 [0, 1]。"""
    rows = []
    if not label_path.is_file():
        return rows
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:5])
            rows.append((cls, cx, cy, w, h))
    return rows


def crop_defect(img, cx, cy, w, h):
    """把归一化的中心宽高换成像素框并裁图。"""
    W, H = img.size
    x1 = int(round((cx - w / 2) * W))
    y1 = int(round((cy - h / 2) * H))
    x2 = int(round((cx + w / 2) * W))
    y2 = int(round((cy + h / 2) * H))
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 - x1 < MIN_CROP_PX or y2 - y1 < MIN_CROP_PX:
        return None
    return img.crop((x1, y1, x2, y2))


def extract_features(args, images_dir, labels_dir):
    """挂 hook 抓检测头前一层的输出，逐个缺陷区域前向一次。"""
    model = YOLO(args.weights)
    net = model.model.to(args.device).eval()

    captured = {}

    def hook(_module, _inp, out):
        captured["feat"] = out

    target_layer = net.model[-2]      # 检测头是 model[-1]，往前一层拿特征
    handle = target_layer.register_forward_hook(hook)

    feats, labels = [], []
    img_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    print(f"{len(img_paths)} 张图待处理")

    try:
        for n, img_path in enumerate(img_paths, 1):
            rows = read_labels(labels_dir / f"{img_path.stem}.txt")
            if not rows:
                continue
            img = Image.open(img_path).convert("RGB")

            for cls, cx, cy, w, h in rows:
                crop = crop_defect(img, cx, cy, w, h)
                if crop is None:
                    continue

                # 缺陷区域尺寸各异，统一缩到 imgsz 再送网络，保证特征维度一致
                crop = crop.resize((args.imgsz, args.imgsz), Image.BILINEAR)
                x = torch.from_numpy(np.asarray(crop)).permute(2, 0, 1).float() / 255.0
                x = x.unsqueeze(0).to(args.device)

                captured.clear()
                with torch.no_grad():
                    net(x)
                if "feat" not in captured:
                    continue

                out = captured["feat"]
                if isinstance(out, (list, tuple)):
                    out = out[-1]
                # (1, C, H, W) 全局平均池化成 (C,)
                vec = out.detach().float().mean(dim=(2, 3)).squeeze(0).cpu().numpy()

                feats.append(vec)
                labels.append(cls)

                if args.max_samples and len(feats) >= args.max_samples:
                    print(f"已达到 max_samples={args.max_samples}，停止采集")
                    return np.stack(feats), np.array(labels)

            if n % 50 == 0:
                print(f"  {n}/{len(img_paths)} 张，已采集 {len(feats)} 个样本")
    finally:
        handle.remove()

    if not feats:
        raise RuntimeError("一个特征都没采到，检查图片和标签是否同名对应")
    return np.stack(feats), np.array(labels)


def reduce_dims(X, args):
    X = StandardScaler().fit_transform(X)

    # PCA 的维度不能超过样本数和原始维度
    n_comp = min(args.pca_dims, X.shape[0], X.shape[1])
    if n_comp < X.shape[1]:
        X = PCA(n_components=n_comp, random_state=args.seed).fit_transform(X)
        print(f"PCA 降到 {n_comp} 维")

    # t-SNE 要求 perplexity < 样本数
    perp = min(args.perplexity, max(5.0, (X.shape[0] - 1) / 3))
    print(f"t-SNE perplexity={perp:.1f}，样本数 {X.shape[0]}")
    return TSNE(n_components=2, perplexity=perp, init="pca",
                random_state=args.seed).fit_transform(X)


def plot(emb, labels, class_names, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
    cmap = plt.get_cmap("tab10")

    for cls in sorted(set(labels.tolist())):
        m = labels == cls
        ax.scatter(emb[m, 0], emb[m, 1], s=14, alpha=0.7,
                   color=cmap(cls % 10),
                   label=f"{cls}_{class_names[cls]} (n={int(m.sum())})")

    ax.set_title("t-SNE Visualization of YOLOv8 Deep Features")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"图已保存到 {out_path}")


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    images_dir, labels_dir, class_names = load_dataset_cfg(args.data, args.split)
    X, y = extract_features(args, images_dir, labels_dir)
    print(f"特征矩阵 {X.shape}，覆盖 {len(set(y.tolist()))} 个类别")

    emb = reduce_dims(X, args)
    plot(emb, y, class_names, args.out)


if __name__ == "__main__":
    main()

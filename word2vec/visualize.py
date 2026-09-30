"""词向量可视化：语义类别散点（PCA / t-SNE）与近邻图。

用法（仓库根目录）：
    conda activate nlp2026
    python word2vec/visualize.py --vectors output/sg_w5.txt
    python word2vec/visualize.py --vectors output/sg_w5.txt --perplexity 10 --suffix _p10
    python word2vec/visualize.py --which hparams          # 超参对比折线图（读 results/）
    python word2vec/visualize.py --which lr               # 学习率曲线（读训练日志）

输出（word2vec/figures/）：
    pca{suffix}.png         语义类别 PCA 散点图
    tsne{suffix}.png        语义类别 t-SNE 散点图
    neighbors{suffix}.png   若干查询词及其 top-10 近邻的 t-SNE 图
    hparams{suffix}.png     窗口大小超参对比折线图
    lr_curve{suffix}.png    学习率随训练进度变化
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from evaluate import load_vectors, top_neighbors

ROOT = Path(__file__).resolve().parents[1]

plt.rcParams["font.sans-serif"] = [
    "WenQuanYi Zen Hei", "Noto Sans CJK SC", "SimHei", "PingFang SC",
]
plt.rcParams["axes.unicode_minus"] = False

CATEGORIES = {
    "地区": ["北京", "上海", "广东", "江苏", "浙江", "山东", "河南", "四川",
             "湖北", "湖南", "福建", "安徽", "河北", "陕西", "辽宁"],
    "动物": ["狗", "猫", "老虎", "狮子", "大象", "熊猫", "兔子", "马",
             "牛", "羊", "鸡", "鸭", "鱼", "鸟", "蛇"],
    "颜色": ["红色", "橙色", "黄色", "绿色", "蓝色", "紫色", "黑色",
             "白色", "灰色", "粉色", "棕色"],
    "朝代": ["秦朝", "汉朝", "唐朝", "宋朝", "元朝", "明朝", "清朝",
             "隋朝", "晋朝", "北魏"],
    "职业": ["医生", "教师", "律师", "警察", "记者", "工程师", "护士",
             "农民", "司机", "厨师"],
    "体育": ["足球", "篮球", "排球", "乒乓球", "羽毛球", "网球", "游泳",
             "跑步", "滑雪", "拳击"],
    "学科": ["数学", "物理", "化学", "生物", "历史", "地理", "哲学",
             "经济", "法律", "医学"],
}

NEIGHBOR_QUERIES = ["北京", "苹果", "足球", "计算机", "孔子", "春节"]


def parse_args():
    p = argparse.ArgumentParser(description="word vector visualization")
    p.add_argument("--vectors", default=str(ROOT / "output/sg_w5.txt"))
    p.add_argument("--fig-dir", default=str(ROOT / "word2vec/figures"))
    p.add_argument("--perplexity", type=float, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--suffix", default="")
    p.add_argument("--which",
                   choices=["all", "pca", "tsne", "neighbors", "hparams", "lr"],
                   default="all")
    p.add_argument("--results-dir", default=str(ROOT / "word2vec/results"))
    p.add_argument("--log", default=str(ROOT / "output/logs/sg_w5.log"))
    return p.parse_args()


def perplexity_for(perplexity, n):
    return float(min(perplexity, max(5.0, (n - 1) / 3.0)))


def tsne_2d(x, perplexity, seed):
    return TSNE(
        n_components=2,
        perplexity=perplexity_for(perplexity, len(x)),
        random_state=seed,
        init="pca",
        learning_rate="auto",
    ).fit_transform(x)


def scatter_by_label(coords, labels, words, title, path):
    plt.figure(figsize=(14, 11))
    for label in sorted(set(labels)):
        idx = [i for i, lab in enumerate(labels) if lab == label]
        plt.scatter(coords[idx, 0], coords[idx, 1], s=40, alpha=0.85, label=label)
    for i, word in enumerate(words):
        plt.annotate(word, (coords[i, 0], coords[i, 1]), fontsize=8, alpha=0.9)
    plt.legend(loc="best", fontsize=10)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def run_categories(words, mat, index, args, fig_dir):
    kept_words, labels = [], []
    for category, members in CATEGORIES.items():
        for word in members:
            if word in index:
                kept_words.append(word)
                labels.append(category)
    missing = [w for members in CATEGORIES.values() for w in members if w not in index]
    print(f"类别词共 {sum(len(v) for v in CATEGORIES.values())} 个，"
          f"在词表中找到 {len(kept_words)} 个，未登录 {len(missing)} 个")
    if len(kept_words) < 50:
        print(f"[警告] 可用类别词不足 50（当前 {len(kept_words)}），请检查词向量词表")
    x = mat[[index[w] for w in kept_words]]

    pca_coords = PCA(n_components=2, random_state=args.seed).fit_transform(x)
    tsne_coords = tsne_2d(x, args.perplexity, args.seed)
    for name, coords in [("PCA", pca_coords), ("t-SNE", tsne_coords)]:
        print(f"{name} 二维坐标上按类别的轮廓系数：{silhouette_score(coords, labels):.3f}")

    if args.which in ("all", "pca"):
        scatter_by_label(pca_coords, labels, kept_words,
                         f"PCA 语义类别（{len(kept_words)} 词）", fig_dir / f"pca{args.suffix}.png")
    if args.which in ("all", "tsne"):
        scatter_by_label(tsne_coords, labels, kept_words,
                         f"t-SNE 语义类别（perplexity={perplexity_for(args.perplexity, len(x)):.0f}）",
                         fig_dir / f"tsne{args.suffix}.png")


def run_neighbors(words, mat, index, args, fig_dir):
    groups, all_words = {}, []
    for query in NEIGHBOR_QUERIES:
        nbrs = top_neighbors(query, index, mat, words, 10)
        if nbrs is None:
            print(f"[警告] 查询词不在词表中：{query}")
            continue
        group = [query] + [w for w, _ in nbrs if w != query]
        groups[query] = group
        for w in group:
            if w not in all_words:
                all_words.append(w)
    if not groups:
        return
    x = mat[[index[w] for w in all_words]]
    coords = tsne_2d(x, args.perplexity, args.seed)
    pos = {w: i for i, w in enumerate(all_words)}

    plt.figure(figsize=(15, 12))
    cmap = plt.get_cmap("tab10")
    for i, (query, group) in enumerate(groups.items()):
        color = cmap(i % 10)
        idx = [pos[w] for w in group]
        plt.scatter(coords[idx, 0], coords[idx, 1], s=45, alpha=0.85,
                    color=color, label=f"{query} 及近邻")
        qi = pos[query]
        plt.scatter(coords[qi, 0], coords[qi, 1], s=260, marker="*", color=color,
                    edgecolors="black", linewidths=0.6, zorder=5)
    for w in all_words:
        plt.annotate(w, (coords[pos[w], 0], coords[pos[w], 1]), fontsize=8, alpha=0.9)
    plt.legend(loc="best", fontsize=10)
    plt.title("查询词（星号）及其 top-10 近邻的 t-SNE 图")
    plt.tight_layout()
    plt.savefig(fig_dir / f"neighbors{args.suffix}.png", dpi=200)
    plt.close()


def run_hparams(args, fig_dir):
    results_dir = Path(args.results_dir)
    points = []
    for path in sorted(results_dir.glob("similarity_sg_w*.json")):
        tag = path.stem[len("similarity_"):]
        m = re.fullmatch(r"sg_w(\d+)", tag)
        if not m:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        points.append((int(m.group(1)), data["datasets"]))
    if not points:
        print(f"[警告] {results_dir} 下没有 similarity_sg_w*.json，无法画超参对比图")
        return
    points.sort()
    windows = [p[0] for p in points]
    plt.figure(figsize=(8, 6))
    for ds in ["wordsim-240.txt", "wordsim-297.txt", "COS960.txt"]:
        ys = [p[1].get(ds, {}).get("spearman") for p in points]
        if all(y is None for y in ys):
            continue
        plt.plot(windows, ys, marker="o", label=ds)
        for x, y in zip(windows, ys):
            if y is not None:
                plt.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                             xytext=(0, 8), fontsize=8, ha="center")
    plt.xlabel("窗口大小 m")
    plt.ylabel("Spearman ρ")
    plt.xticks(windows)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.title("不同窗口大小在 3 个评测集上的结果")
    plt.tight_layout()
    plt.savefig(fig_dir / f"hparams{args.suffix}.png", dpi=200)
    plt.close()
    print(f"超参对比图已保存：{fig_dir / f'hparams{args.suffix}.png'}")


def run_lr_curve(args, fig_dir):
    log_path = Path(args.log)
    if not log_path.exists():
        print(f"[警告] 训练日志不存在：{log_path}")
        return
    pattern = re.compile(r"Alpha: ([0-9.eE+-]+)\s+Progress: ([0-9.]+)%")
    progress, alpha = [], []
    for m in pattern.finditer(log_path.read_text(encoding="utf-8", errors="ignore")):
        alpha.append(float(m.group(1)))
        progress.append(float(m.group(2)))
    if not progress:
        print(f"[警告] {log_path} 中没有 Alpha/Progress 记录")
        return
    step = max(1, len(progress) // 4000)
    plt.figure(figsize=(8, 6))
    plt.plot(progress[::step], alpha[::step], linewidth=1)
    plt.xlabel("训练进度（%）")
    plt.ylabel("学习率 α")
    plt.title(f"学习率随训练进度变化（{log_path.name}）")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / f"lr_curve{args.suffix}.png", dpi=200)
    plt.close()
    print(f"学习率曲线已保存：{fig_dir / f'lr_curve{args.suffix}.png'}")


def main():
    args = parse_args()
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if args.which == "hparams":
        run_hparams(args, fig_dir)
        return
    if args.which == "lr":
        run_lr_curve(args, fig_dir)
        return

    words, mat, index = load_vectors(args.vectors)
    print(f"词向量：{args.vectors}；词表 {len(words)}，维度 {mat.shape[1]}")

    if args.which in ("all", "pca", "tsne"):
        run_categories(words, mat, index, args, fig_dir)
    if args.which in ("all", "neighbors"):
        run_neighbors(words, mat, index, args, fig_dir)
    print(f"图片已写入 {fig_dir}")


if __name__ == "__main__":
    main()

"""词向量的内在评测：词相似度（Spearman）、最近邻检索、COS960 误差分析。

用法（仓库根目录）：
    conda activate nlp2026
    python word2vec/evaluate.py --vectors output/sg_w5.txt

结果写入 word2vec/results/：
    similarity_<tag>.json          3 个数据集的 总对数/覆盖对数/rho
    similarity_<tag>_<ds>.tsv      每个词对的 人工分/模型余弦
    neighbors_<tag>.tsv            查询词的 top-N 最近邻
    cos960_errors_<tag>.tsv        COS960 上模型与人工排序差异最大的词对
其中 <tag> 为词向量文件名（不含扩展名），便于超参对比时区分。
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]

DATASETS = ["wordsim-240.txt", "wordsim-297.txt", "COS960.txt"]

DEFAULT_QUERIES = [
    "北京", "语言", "苹果", "计算机", "足球", "孔子", "春节", "快乐",
    "小米", "银行", "同志", "打",
]


def parse_args():
    p = argparse.ArgumentParser(description="intrinsic evaluation of word vectors")
    p.add_argument("--vectors", default=str(ROOT / "output/sg_w5.txt"))
    p.add_argument("--eval-dir", default=str(ROOT / "data/eval"))
    p.add_argument("--out-dir", default=str(ROOT / "word2vec/results"))
    p.add_argument("--topn", type=int, default=10)
    p.add_argument("--queries", nargs="*", default=DEFAULT_QUERIES)
    return p.parse_args()


def load_vectors(path):
    words, rows = [], []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().split()
        if len(header) != 2:
            raise ValueError(f"{path}: 首行应为 '词数 维度'，实际为 {header!r}")
        dim = int(header[1])
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) < dim + 1:
                continue
            words.append(parts[0])
            rows.append(parts[1: dim + 1])
    mat = np.asarray(rows, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat /= norms
    return words, mat, {w: i for i, w in enumerate(words)}


def read_pairs(path):
    pairs, gold = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                score = float(parts[2])
            except ValueError:
                continue
            pairs.append((parts[0], parts[1]))
            gold.append(score)
    return pairs, gold


def similarity_scores(pairs, gold, index, mat):
    kept_gold, kept_cos, kept = [], [], []
    for (w1, w2), g in zip(pairs, gold):
        i, j = index.get(w1), index.get(w2)
        if i is None or j is None:
            continue
        cos = float(mat[i] @ mat[j])
        kept_gold.append(g)
        kept_cos.append(cos)
        kept.append({"w1": w1, "w2": w2, "gold": g, "cos": cos})
    rho = float(spearmanr(kept_gold, kept_cos).statistic) if len(kept) >= 3 else float("nan")
    return len(pairs), len(kept), rho, kept


def top_neighbors(word, index, mat, words, topn):
    i = index.get(word)
    if i is None:
        return None
    sims = mat @ mat[i]
    sims[i] = -np.inf
    top = np.argpartition(-sims, min(topn, len(sims) - 2))[:topn]
    top = top[np.argsort(-sims[top])]
    return [(words[j], float(sims[j])) for j in top]


def cos960_errors(kept, topn=5):
    if len(kept) < 2:
        return []
    gold = np.array([r["gold"] for r in kept], dtype=np.float64)
    cos = np.array([r["cos"] for r in kept], dtype=np.float64)
    r_model = rankdata(-cos, method="average")
    r_gold = rankdata(-gold, method="average")
    diff = np.abs(r_model - r_gold)
    errors = []
    for k in np.argsort(-diff)[:topn]:
        r = kept[int(k)]
        errors.append({
            "w1": r["w1"], "w2": r["w2"], "gold": r["gold"], "cos": r["cos"],
            "model_rank": int(r_model[k]), "human_rank": int(r_gold[k]),
            "rank_diff": round(float(diff[k]), 1),
        })
    return errors


def main():
    args = parse_args()
    words, mat, index = load_vectors(args.vectors)
    tag = Path(args.vectors).stem
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"词向量：{args.vectors}；词表 {len(words)}，维度 {mat.shape[1]}")

    summary = {"vectors": args.vectors, "vocab_size": len(words),
               "dim": int(mat.shape[1]), "datasets": {}}
    for name in DATASETS:
        path = Path(args.eval_dir) / name
        if not path.exists():
            continue
        pairs, gold = read_pairs(path)
        total, covered, rho, kept = similarity_scores(pairs, gold, index, mat)
        summary["datasets"][name] = {"pairs_total": total, "pairs_covered": covered,
                                     "spearman": round(rho, 4)}
        print(f"{name}: 总对数 {total}，覆盖对数 {covered}，Spearman ρ = {rho:.4f}")
        with open(out_dir / f"similarity_{tag}_{name}.tsv", "w", encoding="utf-8") as f:
            f.write("word1\tword2\tgold\tcosine\n")
            for r in kept:
                f.write(f"{r['w1']}\t{r['w2']}\t{r['gold']}\t{r['cos']:.6f}\n")
        if name == "COS960.txt":
            errors = cos960_errors(kept)
            with open(out_dir / f"cos960_errors_{tag}.tsv", "w", encoding="utf-8") as f:
                f.write("rank_diff\tmodel_rank\thuman_rank\tgold\tcosine\tword1\tword2\n")
                for e in errors:
                    f.write(f"{e['rank_diff']}\t{e['model_rank']}\t{e['human_rank']}\t"
                            f"{e['gold']}\t{e['cos']:.6f}\t{e['w1']}\t{e['w2']}\n")
            print("COS960 模型与人工排序差异最大的 5 对（名次越小越相似）：")
            for e in errors:
                print(f"  {e['w1']}—{e['w2']}：人工 {e['gold']:.2f}（第 {e['human_rank']} 名），"
                      f"余弦 {e['cos']:.3f}（第 {e['model_rank']} 名），差 {e['rank_diff']}")

    with open(out_dir / f"similarity_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(out_dir / f"neighbors_{tag}.tsv", "w", encoding="utf-8") as f:
        f.write("query\trank\tneighbor\tcosine\n")
        for q in args.queries:
            nbrs = top_neighbors(q, index, mat, words, args.topn)
            if nbrs is None:
                print(f"[警告] 查询词不在词表中：{q}")
                continue
            print(f"{q}：" + "、".join(w for w, _ in nbrs))
            for rank, (w, s) in enumerate(nbrs, 1):
                f.write(f"{q}\t{rank}\t{w}\t{s:.6f}\n")

    print(f"结果已写入 {out_dir}")


if __name__ == "__main__":
    main()

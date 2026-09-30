import argparse
import json
import re
import sys
import tempfile
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import jieba
from opencc import OpenCC

ROOT = Path(__file__).resolve().parents[1]

# jieba 默认把词典缓存写到 /tmp/jieba.cache，多用户机器上可能无权限；
# 改写到仓库 output/ 下，避免冲突（output/ 已被 .gitignore 忽略）。
_TMP = ROOT / "output" / "tmp"
_TMP.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(_TMP)

CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
ALNUM_RE = re.compile(r"[a-z0-9]+")


def parse_args():
    p = argparse.ArgumentParser(description="Wikipedia JSON -> segmented corpus.txt")
    p.add_argument("--input", default=str(ROOT / "data/wikipedia-cn-20230720-filtered.json"))
    p.add_argument("--output", default=str(ROOT / "data/corpus.txt"))
    p.add_argument("--eval-dir", default=str(ROOT / "data/eval"))
    p.add_argument("--stats", default=str(ROOT / "word2vec/results/preprocess_stats.json"))
    p.add_argument("--min-count", type=int, default=5)
    p.add_argument("--limit", type=int, default=0, help="only process the first N documents (0 = all)")
    p.add_argument("--workers", type=int, default=16)
    return p.parse_args()


_converter = None


def init_worker():
    global _converter
    _converter = OpenCC("t2s")


def segment(text):
    text = _converter.convert(text).lower()
    tokens = jieba.lcut(text)
    return [t for t in tokens if CJK_RE.fullmatch(t) or ALNUM_RE.fullmatch(t)]


def doc_tokens(record):
    return segment(record.get("completion", ""))


def iter_records(path, limit):
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            for i, record in enumerate(json.load(f)):
                if limit and i >= limit:
                    break
                yield record
        else:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                line = line.strip()
                if line:
                    yield json.loads(line)


def read_eval_pairs(path):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1]))
    return pairs


def eval_coverage(eval_dir, vocab):
    report = {}
    if not Path(eval_dir).is_dir():
        return report
    for path in sorted(Path(eval_dir).glob("*.txt")):
        pairs = read_eval_pairs(path)
        words = {w for pair in pairs for w in pair}
        missing = sorted(w for w in words if w not in vocab)
        covered_pairs = [w1 for w1, w2 in pairs if w1 in vocab and w2 in vocab]
        report[path.name] = {
            "pairs": len(pairs),
            "covered_pairs": len(covered_pairs),
            "unique_words": len(words),
            "covered_words": len(words) - len(missing),
            "word_coverage": round((len(words) - len(missing)) / len(words), 4) if words else 0.0,
            "oov_examples": missing[:40],
        }
    return report


def main():
    args = parse_args()
    started = time.time()

    jieba.initialize()
    counter = Counter()
    documents = 0
    total_tokens = 0

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as out:
        if args.workers > 1:
            pool = Pool(args.workers, initializer=init_worker)
            chunks = pool.imap_unordered(doc_tokens, iter_records(args.input, args.limit), chunksize=64)
        else:
            init_worker()
            chunks = map(doc_tokens, iter_records(args.input, args.limit))
        try:
            for tokens in chunks:
                if not tokens:
                    continue
                out.write(" ".join(tokens) + "\n")
                counter.update(tokens)
                documents += 1
                total_tokens += len(tokens)
        finally:
            if args.workers > 1:
                pool.close()
                pool.join()

    vocab = {w for w, c in counter.items() if c >= args.min_count}
    elapsed = time.time() - started

    stats = {
        "input": args.input,
        "output": args.output,
        "limit": args.limit,
        "documents": documents,
        "tokens": total_tokens,
        "types": len(counter),
        "min_count": args.min_count,
        "vocab_size": len(vocab),
        "elapsed_sec": round(elapsed, 1),
        "eval_coverage": eval_coverage(args.eval_dir, vocab),
    }

    if args.stats:
        Path(args.stats).parent.mkdir(parents=True, exist_ok=True)
        Path(args.stats).write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({k: v for k, v in stats.items() if k != "eval_coverage"}, ensure_ascii=False, indent=2))
    for name, cov in stats["eval_coverage"].items():
        print(f"{name}: 唯一词 {cov['covered_words']}/{cov['unique_words']} "
              f"覆盖（{cov['word_coverage']:.2%}）；"
              f"词对 {cov['covered_pairs']}/{cov['pairs']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

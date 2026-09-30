#!/usr/bin/env bash
# 实验四 2(b)：下载/编译官方 word2vec，并训练 skip-gram + 负采样词向量。
# 用法（仓库根目录）：
#   bash word2vec/train.sh prepare   # 只下载并编译官方 C 代码
#   bash word2vec/train.sh baseline  # 基线：sg_w5 (d=200, m=5, K=5)
#   bash word2vec/train.sh window    # 超参对比：窗口 m in {2, 5, 10}
#   bash word2vec/train.sh dim       # 超参对比：维度 d in {50, 100, 200, 300}
#   bash word2vec/train.sh all       # 默认 = window
# 已存在 output/<name>.txt 时自动跳过，不重复训练；FORCE=1 可强制重训。
set -euo pipefail
cd "$(dirname "$0")/.."

W2V_REPO="https://github.com/tmikolov/word2vec.git"
W2V_COMMIT="20c129a"
W2V_DIR="output/word2vec_official"
BIN="$W2V_DIR/word2vec"
CORPUS="${CORPUS:-data/corpus.txt}"
THREADS="${THREADS:-8}"
LOG_DIR="output/logs"

if [ ! -d "$W2V_DIR/.git" ]; then
  git clone "$W2V_REPO" "$W2V_DIR"
fi
git -C "$W2V_DIR" checkout --quiet "$W2V_COMMIT"

if [ "$(uname -s)" = "Darwin" ]; then
  # macOS 编译官方代码会报 fgetc_unlocked 未声明，按作业提示替换
  make -C "$W2V_DIR" CFLAGS="-O3 -Wall -funroll-loops -Wno-unused-result -Dfgetc_unlocked=getc_unlocked"
else
  make -C "$W2V_DIR"
fi
echo "官方 word2vec 就绪：$BIN（commit $W2V_COMMIT）"

train() {  # train <输出名> <窗口> <维度>；其余超参固定
  local name="$1" window="$2" size="$3"
  if [ -s "output/$name.txt" ] && [ "${FORCE:-0}" != "1" ]; then
    echo ">>> 跳过 $name：output/$name.txt 已存在（FORCE=1 bash word2vec/train.sh ... 可强制重训）"
    return 0
  fi
  echo ">>> $name: size=$size window=$window negative=5 hs=0 sample=1e-4 min-count=5 iter=5 threads=$THREADS"
  local t0 t1
  t0=$(date +%s)
  "$BIN" -train "$CORPUS" -output "output/$name.txt" \
    -cbow 0 -size "$size" -window "$window" -negative 5 -hs 0 \
    -sample 1e-4 -min-count 5 -iter 5 -threads "$THREADS" -binary 0 \
    > "$LOG_DIR/$name.log" 2>&1
  t1=$(date +%s)
  echo "$((t1 - t0))" > "$LOG_DIR/$name.time"
  echo ">>> $name 完成，耗时 $((t1 - t0)) 秒"
}

case "${1:-all}" in
  prepare)
    ;;
  baseline)
    [ -f "$CORPUS" ] || { echo "缺少 $CORPUS，请先运行 word2vec/preprocess.py" >&2; exit 1; }
    mkdir -p output "$LOG_DIR"
    train sg_w5 5 200
    ;;
  window|all)
    [ -f "$CORPUS" ] || { echo "缺少 $CORPUS，请先运行 word2vec/preprocess.py" >&2; exit 1; }
    mkdir -p output "$LOG_DIR"
    for w in 2 5 10; do
      train "sg_w${w}" "$w" 200
    done
    ;;
  dim)
    [ -f "$CORPUS" ] || { echo "缺少 $CORPUS，请先运行 word2vec/preprocess.py" >&2; exit 1; }
    mkdir -p output "$LOG_DIR"
    for d in 50 100 200 300; do
      train "sg_d${d}" 5 "$d"
    done
    ;;
  *)
    echo "未知参数：$1（可选 prepare | baseline | window | dim | all）" >&2
    exit 1
    ;;
esac

echo "完成。词向量在 output/，训练日志在 $LOG_DIR/"

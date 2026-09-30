# NLP 2026 实验四：词向量

- 姓名：李贵龙
- 学号：202411581243

word2vec 的梯度推导，以及在中文维基百科语料上训练、评测与可视化中文词向量。

## 环境

- Linux（Intel Xeon Silver 4210R，40 核，62 GB 内存）
- conda 环境 `nlp2026`（Python 3.10）
- 依赖：

```bash
conda activate nlp2026
pip install numpy scipy matplotlib scikit-learn jieba opencc-python-reimplemented
```

- 训练路线 A：官方 C 代码 [tmikolov/word2vec](https://github.com/tmikolov/word2vec)，
  commit `20c129a`（由 `word2vec/train.sh` 自动 clone 到 `output/`，不拷贝源码进仓库）。

## 数据准备

原始语料与评测集放在 `data/`（不提交 Git）：

```bash
mkdir -p data && cd data
wget https://hf-mirror.com/datasets/pleisto/wikipedia-cn-20230720-filtered/resolve/main/wikipedia-cn-20230720-filtered.json
```

评测集（课程下发）放 `data/eval/`：

```text
data/eval/wordsim-240.txt   # 240 对，相关性
data/eval/wordsim-297.txt   # 297 对，相关性
data/eval/COS960.txt        # 960 对，近义程度（第 3 列为平均分）
```

## 如何复现

```bash
conda activate nlp2026

# 1. 预处理：JSON -> 清洗 -> 繁转简 -> jieba 分词 -> data/corpus.txt
python word2vec/preprocess.py

# 2. 下载/编译官方 word2vec 并训练（基线 + 窗口对比 m in {2,5,10}）
bash word2vec/train.sh window

# 3. 评测各设置词向量（相似度 / 最近邻 / COS960 误差分析）
for v in output/sg_w2.txt output/sg_w5.txt output/sg_w10.txt; do
    python word2vec/evaluate.py --vectors "$v"
done

# 4. 可视化（pca.png / tsne.png / neighbors.png / hparams.png / lr_curve.png）
python word2vec/visualize.py --vectors output/sg_w5.txt
python word2vec/visualize.py --which hparams   # 超参对比折线图（读 results/similarity_sg_w*.json）
python word2vec/visualize.py --which lr        # 学习率曲线（读 output/logs/sg_w5.log）
```

基线超参数：skip-gram + 负采样，`d=200, m=5, K=5, sample=1e-4, min_count=5, iter=5, threads=8`。
训练日志（含学习率进度）在 `output/logs/`；各超参设置的评测结果在 `word2vec/results/`，
图片在 `word2vec/figures/`，报告中的数字均取自这些文件。

## 目录结构

```text
nlp-fall2026-lab4/
|-- README.md
|-- .gitignore
|-- data/                     # 原始语料、分词语料、评测集（不提交）
|-- output/                   # 词向量、日志、官方源码（不提交）
|-- word2vec/
|   |-- preprocess.py         # 预处理
|   |-- train.sh              # 下载/编译/训练
|   |-- evaluate.py           # 相似度、最近邻、误差分析
|   |-- visualize.py          # PCA / t-SNE / 近邻图
|   |-- results/              # 评测结果（提交）
|   `-- figures/              # 图片（提交）
`-- report/
    |-- report.tex
    |-- refs.bib
    `-- report.pdf
```

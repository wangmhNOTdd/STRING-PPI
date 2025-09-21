# PPI Prediction on STRING (Human) with ESM2 + HGCN

> 基于 STRING v12.0 的 **Homo sapiens (9606)** 物理互作子集，使用 **ESM2** 生成蛋白质序列表征，再用 **Hyperbolic GCN (HGCN)** 进行链路预测（PPI 预测）。本 README 提供完整项目规划、数据处理流程、训练与评测指南。

---

## 1. Project Goals 项目目标

- 从 STRING v12.0 人类 **physical links** 构建高质量 PPI 图（建议 `combined_score >= 700` 或 `>= 900`）。  
- 使用 **ESM2** 将蛋白质序列编码为高维特征。  
- 在**双曲空间**中用 **HGCN** 学习图的层级结构，并进行**链路预测**。  
- 提供可复现实验脚本、指标（AUROC/AUPRC/F1）、消融与可视化。

---

## 2. Repo Structure 目录结构（当前仓库）

```
.
├─ README_PPI_ESM2_HGCN.md
├─ configs/
│  ├─ data.yaml
│  ├─ esm2.yaml
│  ├─ model.yaml
│  └─ train.yaml
├─ data/
│  ├─ SHS27K/                      # 另一个数据集（如不使用可忽略）
│  └─ string/
│     ├─ raw/                      # 已下载的 STRING v12.0 人类 9606 文件
│     ├─ interim/                  # 预处理输出：nodes.tsv / edges.tsv / seqs.fasta
│     └─ processed/                # 最终图与特征：esm2_features.npz / graph.pt / node_id_to_idx.npy
├─ models/
│  ├─ decoders/
│  ├─ layers/
│  └─ manifolds/
├─ runs/
│  └─ hgcn_esm2_700/
│     └─ best.pt                   # 训练保存的最佳权重
└─ src/
  ├─ build_graph.py               # 构图、负采样、划分并输出 graph.pt
  ├─ download_string.py           # 占位（当前为空）
  ├─ embed_esm2.py                # 生成 ESM2 表征并保存 .npz
  ├─ evaluate.py                  # 占位（当前为空）
  ├─ preprocess_string.py         # 过滤、ID 映射、序列对齐
  ├─ train.py                     # 训练入口（含简单评估与保存 best.pt）
  └─ utils.py
```

---

## 3. Data 数据说明

**来源**：STRING Database v12.0  
- 主页：https://string-db.org/  
- 下载镜像：https://stringdb-downloads.org/

**人类 (9606) 文件**：
- `protein.physical.links.v12.0.txt.gz`：物理互作边（包含 `combined_score` 0–1000）  
- `protein.info.v12.0.txt.gz`：蛋白信息（`protein_external_id`、`preferred_name` 等）  
- `protein.aliases.v12.0.txt.gz`：别名 -> 统一 ID 的映射（推荐用于 UniProt/Ensembl 对齐）  
- `protein.sequences.v12.0.fa.gz`：蛋白序列（FASTA，header 通常含 `9606.ENSP...`）

**使用建议**：
1) 使用 `combined_score >= 700`（高置信）或 `>= 900`（超高置信）。  
2) 用 `protein.aliases` 做 ID 映射，统一到与序列一致的 ID（例如 Ensembl Protein ID）。  
3) 移除自环、重复边、多重边；仅保留图中存在序列的蛋白。  
4) 全部文件均为 gzip 压缩。

---

## 4. Environment 环境与依赖

最小依赖（当前代码已验证）
- Python 3.9+
- pytorch + torch-geometric（按官方指引安装相应 CUDA 版本）
- transformers, tokenizers
- biopython, pandas, numpy, scikit-learn, tqdm

示例（CPU，仅示意，具体以你本机/显卡选择为准）
```powershell
pip install pandas numpy scikit-learn tqdm biopython transformers
# 按 https://pytorch-geometric.readthedocs.io/ 的说明安装 torch 和 pyg
```
---

## 5. Data Pipeline 数据流水线

### 5.1 下载
已下载至data/string/

### 5.2 预处理与对齐（过滤 + ID 映射 + 序列对齐）
```bash
python src/preprocess_string.py \
  --raw_dir data/string/raw \
  --interim_dir data/string/interim \
  --min_score 700 \
  --map_to preferred_name \
  --drop_no_sequence
```
关键步骤：
- 过滤 `combined_score`；
- 通过 `protein.aliases` 统一到选择的 ID（建议与 FASTA header 一致，如 `9606.ENSP...`）；
- 仅保留在 `protein.sequences` 中存在序列的节点；
- 去重、自环清理、保证无多重边；
- 输出对齐后的 `nodes.tsv`, `edges.tsv` 与 `seqs.fasta`。

### 5.3 生成 ESM2 序列表征
```bash
python src/embed_esm2.py \
  --fasta data/string/interim/seqs.fasta \
  --esm2_variant esm2_t12_35M_UR50D \
  --batch_size 16 \
  --pooling mean \
  --device cuda \
  --out_npz data/string/processed/esm2_features.npz
```
- 支持 `--pooling mean|cls`；建议资源有限用 `esm2_t12_35M_UR50D`。
- 输出 `esm2_features.npz`，包含 `keys`（序列 ID 顺序）与 `embeddings`（对齐矩阵）。

### 5.4 构图与负采样 + 数据划分
```bash
python src/build_graph.py \
  --nodes data/string/interim/nodes.tsv \
  --edges data/string/interim/edges.tsv \
  --features data/string/processed/esm2_features.npz \
  --neg_ratio 1 \
  --split transductive \
  --val_ratio 0.1 \
  --test_ratio 0.2 \
  --out_dir data/string/processed
```
- **负采样**：默认按 1:1 负样本，支持
  - **随机非边采样**（默认）
  - **度感知采样**（控制假阴性率）  
- **划分策略**：
  - `transductive`：所有节点可见，仅边划分（常见）；
  - `inductive`：留出部分节点（其相关边为 val/test），考察模型在新蛋白上的泛化。

生成文件：`data/string/processed/graph.pt` 与 `node_id_to_idx.npy`。
注意：划分 mask 已包含在 `graph.pt` 中（`train_mask/val_mask/test_mask`），不再单独导出 `splits.npz`。

---

## 6. Model 模型设计（HGCN in Poincaré Ball）

### 6.1 输入与嵌入
- 初始特征：ESM2 向量 `x ∈ R^d`；
- 欧氏 -> 双曲：`x_h = exp_map0(Linear(x))`；
- 加入节点度、中心性等结构特征（同样通过 `exp_map0` 投入双曲空间）。

### 6.2 HGCN 层
- 使用 Möbius 运算（加法/仿射/矩阵乘）与 Riemannian 优化；
- 消息传递：在切空间聚合，再映回双曲空间（或使用双曲聚合算子）；
- 可学习**曲率 c**（或固定），层与层之间共享/独立均可配置；
- 正则：范数约束、球内边界约束（避免数值爆炸）。

### 6.3 解码器（链路预测）
- **双曲距离解码**：
  - 对结点对 \(u,v\)，以 Poincaré 距离 \(d_{\mathbb{B}}(u,v)\) 作为相似度的负项：  
    \( s(u,v) = -d_{\mathbb{B}}(u,v) \)，再经 `sigmoid(s/τ)` 得到概率；
- 备选：
  - 切空间内 DistMult/MLP；
  - 双曲内积（Lorentz 模式）。

### 6.4 损失与训练
- **Binary Cross Entropy (BCE)** 或 **infoNCE**（带温度、边/非边对比）；
- 负采样 on-the-fly；
- Riemannian Adam (via `geoopt.optim.RiemannianAdam`)；
- 混合精度、梯度裁剪、早停。

---

## 7. Training & Evaluation 训练与评测

### 7.1 训练
```bash
python src/train.py \
  --config configs/model.yaml \
  --data_dir data/string/processed \
  --epochs 100 \
  --lr 1e-3 \
  --batch_size 8192 \
  --model hgcn \
  --decoder hyp_distance \
  --log_dir runs/hgcn_esm2_700
```
视情况调整

### 7.2 评测
- 当前 `src/evaluate.py` 为占位空文件；训练脚本已在每个 epoch 输出验证/测试集损失与准确率，并将 `val_loss` 最优权重保存到 `runs/.../best.pt`。
- 如需更丰富指标（AUROC/AUPRC/F1），后续可在 `evaluate.py` 中补充；或在 `train.py` 的 `evaluate_model` 基础上扩展 sklearn/torchmetrics。


---

## 8. Pitfalls & Tips 注意事项

- **ID 对齐**：确保边两端的 ID 与序列 FASTA 中的 ID 完全一致；  
- **假阴性**：STRING 未覆盖的真实互作会被当作负样本，度感知/近邻排除可降低影响；  
- **数据泄露**：划分时避免把同一蛋白的多条高置信互作分别落在 train/test 造成“过度容易”；可做 **protein-level split** 以更严格；  
- **数值稳定**：双曲运算靠近边界时需裁剪（`||x|| < 1 - eps`），并用稳定的 `artanh` 实现；  
- **资源**：ESM2 大模型推理分批 + AMP，必要时缓存到磁盘并开启内存映射。

---

## 9. Minimal Config 最小可运行配置示例

`configs/data.yaml`
```yaml
species: 9606
min_score: 700
paths:
  raw: data/string/raw
  interim: data/string/interim
  processed: data/string/processed
mapping:
  prefer_id: ensp         # 或 preferred_name / uniprot，需与 FASTA 对齐
dedup: true
drop_no_sequence: true
```

`configs/esm2.yaml`
```yaml
variant: esm2_t12_35M_UR50D   # 嵌入脚本不直接读取该文件，仅供参考
pooling: mean                 # or cls
batch_size: 16
device: cuda
```

`configs/model.yaml`
```yaml
model: hgcn
manifold: poincare
dim: 128
layers: 2
curvature:
  learnable: true
  init: 1.0
dropout: 0.1
decoder: hyp_distance
```

`configs/train.yaml`
```yaml
epochs: 100
lr: 0.001
batch_size: 8192
neg_ratio: 1
split: transductive
val_ratio: 0.1
test_ratio: 0.2
seed: 42
```
说明：当前代码仅在 `train.py` 中读取 `configs/model.yaml` 的少量键（如 `dim`）；其余 YAML 文件为参考配置，未被脚本自动消费。

---

## 10. CLI Examples 命令汇总

```bash
# 1) Download（已下载可跳过；当前 download_string.py 为空占位）
# 手动从 STRING 下载人类 9606 所需 4 个文件到 data/string/raw/
# https://stringdb-downloads.org/ （v12.0）

# 2) Preprocess
python src/preprocess_string.py --raw_dir data/string/raw --interim_dir data/string/interim --min_score 900 --map_to ensp --drop_no_sequence

# 3) ESM2 embeddings
python src/embed_esm2.py --fasta data/string/interim/seqs.fasta --esm2_variant esm2_t12_35M_UR50D --batch_size 16 --pooling mean --device cuda --out_npz data/string/processed/esm2_features.npz

# 4) Build graph & splits
python src/build_graph.py --nodes data/string/interim/nodes.tsv --edges data/string/interim/edges.tsv --features data/string/processed/esm2_features.npz --neg_ratio 1 --split transductive --val_ratio 0.1 --test_ratio 0.2 --out_dir data/string/processed

# 5) Train HGCN
python src/train.py --config configs/model.yaml --data_dir data/string/processed --epochs 100 --lr 1e-3 --batch_size 8192 --model hgcn --decoder hyp_distance --log_dir runs/hgcn_esm2_700
# 6) Evaluate（占位）
# 评测脚本待补充，目前以训练日志与 best.pt 为准
```

---

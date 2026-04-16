# QClip

QClip 是一个端到端的视频向量检索系统，涵盖 **数据预处理 → 特征推理 → 数据打包 → 模型训练 → 向量检索 → 指标评估** 的完整流程。核心思路是：对视频抽帧并提取帧级特征（DINOv3），再通过 learnable query cross-attention 将变长帧序列聚合为定长 slot 特征，最终用于大规模视频检索。

---

## 目录结构

```
qclip/
├── qclip/                   # QClip 训练（数据/模型/loss/配置/训练入口）
│   ├── train.py             # Lightning 训练入口
│   ├── utils.py             # 配置合并 + 模型构建/加载
│   ├── configs/             # YAML 实验配置
│   ├── data/                # WebDataset 数据读取 + partition 采样
│   ├── models/              # QClip 模型结构
│   └── loss/                # InfoNCE / KoLeo 等 loss
├── vrl/                     # VRL 训练（结构与 qclip 类似）
├── dinov3/                  # DINOv3 训练
├── data/                    # 离线数据流水线
│   ├── preprocess/          # 抽帧 + 质量过滤 + 去重
│   ├── infer/               # DINOv3 帧特征 / QClip·VRL 聚合特征推理
│   ├── build_wds/           # 打包 WebDataset tar
│   └── aux/                 # 辅助信息计算（bbox / resize_size 等）
├── eval/                    # 向量检索 + 召回指标计算
├── tools/                   # 工具脚本（任务列表扫描/合并、embedding 分片等）
├── scripts/                 # 启动脚本
├── service/                 # 推理服务
└── requirements.txt         # 环境依赖
```

---

## 端到端流程

```
preprocess → infer (DINO) → build_wds → train (QClip) → infer (QClip) → faiss_retrieval → cal_metrics
```

---

## 1. Preprocess：抽帧 + 过滤去重 + 产出 Meta

预处理阶段负责从原始视频中抽帧，并进行可选的质量过滤与去重，最终产出帧级 meta 文件。

### 1.1 视频路径列表模式

| 项 | 说明 |
|---|---|
| **入口** | `data/preprocess/process_video.py` |
| **输入** | txt 文件，每行一个本地视频路径 |
| **输出** | `output_frames_dir/<video_id>/frame_*.jpg`（帧图像）<br>`output_meta_dir/<video_id>.jsonl`（帧级 meta，每行含 `img_path / frame_idx / timestamp / fps` 等） |

**关键逻辑：**

- **抽帧**：优先使用 decord，失败时 fallback 到 OpenCV。
- **质量过滤**（可选）：
  - **PixelEntropy**：基于边缘密度 + 信息熵的质量评分过滤。
  - **PHash**：感知哈希（CUDA 加速），配合 FAISS BinaryFlat + 图连通分量做近似去重。
  - **SSCD Embedding**：语义级去重（TorchScript 模型），配合 FAISS FlatIP + 图连通分量。
- **HDFS 上传**（可选）：帧目录整体上传至 HDFS，并将 meta 中的 `img_path` 替换为 HDFS 路径。

### 1.2 CSV（Album 聚合）模式

| 项 | 说明 |
|---|---|
| **入口** | `data/preprocess/process_album.py` |
| **输入** | CSV 文件（含 `group_id / album_id / book_title / chapter_rank` 等列） |
| **输出** | 每个 `album_id.jsonl`（按 album 聚合的帧级 meta） |

该模式会按 `group_id` 下载视频（支持 m3u8 与直链，必要时进行 AV1 转码），然后走与上述相同的抽帧 + 过滤去重流程。

---

## 2. Infer：特征推理

### 2.1 DINOv3 帧级特征

从 preprocess 产出的 meta jsonl 中读取 `img_path`，逐帧提取 DINOv3 CLS embedding。

| 项 | 说明 |
|---|---|
| **入口** | `data/infer/infer_dino_pretrain.py` |
| **输入** | `--task-jsonl`：任务列表 JSONL，每行至少含 `jsonl_path`（指向 preprocess 的 meta 文件），可选 `bbox / resize_size` |
| **输出** | `output_dir/<stem>.npy`（`stem = Path(jsonl_path).stem`，每帧一个向量） |

**处理流程：**

1. `ImageDataset` 使用 `fsspec` 读取图像（支持本地与 HDFS），执行 crop + resize + 标准化变换。
2. 加载 DINOv3 预训练权重（默认从 `dinov3/pretrained/short_drama`）。
3. `autocast + inference_mode` 推理，输出拼接为 numpy 数组。

**TOS 变体：**

| 入口 | 数据格式 | 输出 |
|---|---|---|
| `infer_dino_pretrain_tos.py` | TOS 上的 `*_frames.zip`（jpg 帧包） | `emb/<album_id>.npy` + `emb_info/<album_id>.jsonl` |
| `infer_dino_pretrain_tos_base64.py` | TOS 上的 JSON 对象（含 base64 图片） | `<name>.npy` + `<name>.jsonl` |

> TOS 变体会额外产出 `emb_info` / 配套 `.jsonl`，确保帧与 embedding 严格对齐。

### 2.2 QClip 聚合特征

将帧级特征聚合为 clip-slot 级特征。

| 项 | 说明 |
|---|---|
| **入口** | `data/infer/infer_qclip.py` |
| **输入** | `--frame-emb-dir`（DINO 帧级 `.npy`）+ `--ckpt-dir`（QClip checkpoint）+ `--task-jsonl`（视频列表） |
| **输出** | `<name>.npy`（`[num_clips * num_slots, D]`，L2 normalized）+ `<name>.jsonl`（每条向量的 `entity_id / clip_id / slot_id` 映射） |

**关键逻辑：**

- `sample_clips`：将变长帧序列切为定长 clip（默认 256 帧/clip，最后一段对齐到末尾）。
- 模型通过 `build_qclip_model_for_eval` 加载 config + state_dict。

### 2.3 VRL 聚合特征

| 项 | 说明 |
|---|---|
| **入口** | `data/infer/infer_vrl.py` |
| **输出** | 与 QClip 类似，但映射仅含 `entity_id / clip_id`（无 `slot_id`） |

---

## 3. Build WDS：打包 WebDataset

将特征与 meta 对齐后打包为 WebDataset tar 格式，供训练阶段流式读取。

### 3.1 QClip 训练用 Embedding WDS

| 项 | 说明 |
|---|---|
| **入口** | `data/build_wds/build_wds_qclip.py` |
| **输入** | `--task-txt`（meta jsonl 路径列表）+ `--emb-dir`（DINO 帧级 `.npy`）+ `--output-dir`（HDFS 目标）+ `--output-prefix`（分片前缀） |
| **输出** | HDFS 上的 tar 分片，每个 sample 包含：`.npy`（clip embedding）、`.jsonl`（对齐的 meta 子片段）、`.json`（`album_id / pair_id`） |

**关键逻辑：**

- `traverse_clips`：50% overlap 滑窗切片（clip_len=512，stride=256）。
- `ShardWriter`：先写本地临时 tar，再通过 `fsspec` 上传至 HDFS，支持 retry + backoff。
- **对齐校验**：`len(embeddings) == len(meta_infos)` 不满足时跳过该实体。

### 3.2 DINOv3 训练用 图像级 WDS

| 项 | 说明 |
|---|---|
| **入口** | `data/build_wds/build_wds_dino.py` |
| **输出** | tar 内含 `.jpg` + `.json`，更通用的图像 dataset 打包格式 |

---

## 4. Train：QClip 模型训练

| 项 | 说明 |
|---|---|
| **入口** | `qclip/train.py` |
| **输入** | `-c/--config`（YAML 配置）、`--log-dir` + `--exp`（输出目录） |
| **输出** | `exp_dir/config.yaml`（落盘配置）+ Lightning checkpoint（`exp_dir/{step}.ckpt`） |

### 4.1 数据加载

1. **DataModule**（`qclip/data/from_config.py` 的 `DataModuleFromConfig`）通过配置动态实例化 dataset。
2. **Dataset**（`qclip/data/toutiao_video_embs.py` 的 `ToutiaoVideoEmbs`）：
   - 使用 `fsspec` 枚举 HDFS 上的 tar shards。
   - 通过 `WebDataset` 以 `pipe:hdfs dfs -cat /<path>` 流式读取。
   - 每个样本读取 `sample["npy"]`（clip embedding）和 `sample["json"]`（`album_id / pair_id`）。
   - `FramesPartition` 在 interval 内随机采样，构造训练 pairs（形状 `[num_parts, part_size, D]`）。

### 4.2 模型结构与训练目标

**模型**（`qclip/models/qclip.py` 的 `QClip`）：

- **输入**：pairs 中的两段 clip（global / local），每段为帧级 CLS embedding 序列。
- **Forward**：帧特征投影到 `embed_dim`，通过 learnable query 做 cross-attention 读出 slot 特征，输出 `[B, query_num, D]`。
- **Training Step**：
  1. 随机选取 `local_clip_len` 截断 local clip。
  2. 分别计算 global / local 两路 slot 特征，stack 为 `[B, 2, N, D]`。
  3. Loss = **KoLeoLoss** + **InfoNCELoss**。
  4. 手动优化（`automatic_optimization=False`）：使用 timm optimizer + scheduler step_update。

---

## 5. Faiss Retrieval：向量检索

| 项 | 说明 |
|---|---|
| **入口** | `eval/faiss_retrieval.py` |
| **输入** | `work_dir/index_merged/index_*.npy + .jsonl`（索引库）、`work_dir/query_merged/query_*.npy + .jsonl`（查询集） |
| **输出** | `work_dir/result/retrieval_results_*.jsonl`（每行含 `q_record / tgt_record / score`） |

**处理流程：**

1. `load_shards`：读取多 shard，可选 L2 normalize（默认开启）。
2. `build_index`：`faiss.index_factory(dim, "Flat" 或 "PCAW{pca_dim},Flat", METRIC_INNER_PRODUCT)`，支持多卡 GPU shard，支持 PCA 白化处理。
3. `search_batch`：分 batch 调用 `index.search`。
4. `save_results`：写出检索结果 jsonl。

> **前置步骤**：通常需要先用 `tools/eval_merge_tuple.py` 将零散的 `<entity>.npy + .jsonl` 合并为按 `shard_size` 切分的大 shard（`index_000000.npy/jsonl`），并保证 `.npy` 行数与 `.jsonl` 行数对齐。

---

## 6. Cal Metrics：召回指标计算

| 项 | 说明 |
|---|---|
| **入口** | `eval/cal_metrics.py` |
| **输入** | `--result-path`（检索结果 jsonl）+ `--gt-path`（GT 映射 jsonl） |
| **输出** | `metrics.json`（Recall@K）+ `error_records.jsonl`（badcase，含 top5 预测） |

**关键参数：**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `query_id_key` | `gid` | 从 `q_record` 中提取 query id 的字段 |
| `target_id_key` | `album_id` | 从 `tgt_record` 中提取 target id 的字段 |
| `gt_query_key` | `gid` | GT 行中 query id 字段 |
| `gt_label_key` | `entity_id` | GT 行中 label id 字段 |

**关键逻辑：**

- `aggregate_scores`：同一 query 的检索结果按 target 聚合（累加 score，支持 `min_score` 过滤）。
- `compute_recall`：对每个 query 排序后计算 Recall@{1, 5, 10}。

---

## 关键约束：文件命名与对齐

整个流水线依赖严格的文件命名与行数对齐，不满足时会导致数据丢失或流程中断：

| 约束 | 说明 |
|---|---|
| **文件名一致** | preprocess 产出的 `meta/<name>.jsonl` 的 `name` 必须与 DINO infer 输出的 `<name>.npy` 保持同一 stem。DINO infer 默认以 `Path(jsonl_path).stem` 命名输出文件，build_wds 也以同一 stem 查找 embedding。 |
| **行数严格对齐** | `build_wds_qclip` 要求 `len(embeddings) == len(meta_infos)`，不一致时该实体会被跳过。 |
| **帧丢弃的一致性** | `infer_dino_pretrain.py` 仅保存 features 不回写 meta；若推理阶段丢帧，meta 与 embedding 会不对齐。`infer_dino_pretrain_tos.py` 因为元数据在TOS上，会额外写出 `emb_info` 避免此问题。 |

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. merlin 上运行入口指令示例
cd /mnt/bn/yuanliang-llm-gpu-train/jiajun/code/qclip && ./scripts/data_preprocess_video.sh ${ARNOLD_ID}

```

# 流水线指南

[返回 README](../README.zh-CN.md) · [English](REPRODUCING.md) | **简体中文**

以下命令均在仓库根目录的 Linux shell 中运行。将 `/path/to/...` 和 HDFS 路径替换为实际位置。仓库未附带权重和数据；推理前需准备相互兼容的帧编码器与 Q-Clip 权重。

## 环境

使用 Python 3.11 和支持 CUDA 的 PyTorch 环境。完整预处理与训练流水线面向 Linux x86_64。除[核心模型环境](../README.zh-CN.md#快速开始)外，还需要：

- 与 CUDA 运行时兼容的 GPU FAISS。`requirements.txt` 固定了 Linux x86_64、CPython 3.11 的 wheel，不能直接用于所有平台。
- CUDA toolkit、nvJPEG、C++17 编译器和 CMake ≥3.18，用于编译本地 `phash_cuda` 扩展。
- 可用的 HDFS 客户端（`hdfs dfs`）、libhdfs，以及配套的 PyArrow/fsspec 配置，用于训练和 WebDataset 打包。
- 使用对应可选工具时所需的 ModelScope/TOS 依赖。完整依赖列表包含 `bytedtos`。

确认上述前提后再安装完整环境：

```bash
python -m pip install -r requirements.txt
# 流水线直接使用、但上述列表未显式列出的依赖：
python -m pip install omegaconf opencv-python pyarrow
```

`scripts/` 中的启动脚本含有集群专用路径和环境变量。本指南直接使用模块入口，无需这些启动脚本。

## 数据约定

视频 ID 需唯一，并保留帧的顺序。帧元数据、帧特征和片段特征使用相同的文件名（不含扩展名）：

| 文件 | 格式 |
| --- | --- |
| `meta/123.jsonl` | 每行一帧保留帧的元数据，包含 `img_path` |
| `frame_embeddings/123.npy` | Float32 数组 `[num_frames, 1536]`，与元数据顺序一致 |
| `clip_embeddings/123.npy` | 归一化数组 `[num_clips × num_slots, 1536]` |
| `clip_embeddings/123.jsonl` | 每行对应一个描述子 |

帧元数据和任务记录示例（分别保存在不同 JSONL 文件中）：

```json
{"video_id":"123","img_path":"/path/to/frames/123/frame_0000.jpg","frame_idx":0,"timestamp":0.0,"fps":25.0}
```

```json
{"jsonl_path":"/path/to/meta/123.jsonl"}
```

特征提取使用任务 JSONL。WebDataset 打包使用 TXT 文件，每行是一份帧元数据 JSONL 的路径。训练视频 ID 需为数字：读取器会将 `album_id` 和 `pair_id` 转为整数。

除非训练所用编码器或权重另有要求，聚合前不要对帧特征做 L2 归一化。帧编码器、特征维度和 checkpoint 必须匹配；论文使用 DINOv2，本次代码快照包含 DINOv3 工具。

## 1. 抽帧与过滤

准备 `videos.txt`，每行一个本地视频路径，然后运行：

```bash
python -m data.preprocess.process_video \
  --input /path/to/videos.txt \
  --output_frames_dir /path/to/frames \
  --output_meta_dir /path/to/meta \
  --target_fps 1
```

默认开启质量过滤、pHash 和 SSCD 去重。SSCD 权重路径为 `data/preprocess/modules/deduplicate/pretrained/sscd_disc_mixup.torchscript.pt`。可用 `--no-sscd`、`--no-phash`、`--no-entropy` 关闭对应阶段；模块导入所需依赖仍需安装。此入口逐视频处理，跨整个语料库去重需另行组织和处理数据。

## 2. 提取帧特征

`infer_dino_pretrain` 读取 `dinov3/pretrained/short_drama/config.yaml` 和 `teacher_checkpoint.pth`。需先准备兼容骨干网络的文件；当前路径写在 `build_default_model()` 中。

```bash
python -m data.infer.infer_dino_pretrain \
  --task-jsonl /path/to/tasks.jsonl \
  --output-dir /path/to/frame_embeddings \
  --batch-size 128 --num-workers 8
```

输出文件为 `<元数据文件名>.npy`。图像读取失败时会跳过对应帧，但不会重写原始元数据。打包前需处理失败帧，保证元数据与特征逐行严格对齐。

## 3. 打包训练分片

```bash
python -m data.build_wds.build_wds_qclip \
  --task-txt /path/to/meta_paths.txt \
  --emb-dir /path/to/frame_embeddings \
  --output-dir /user/you/qclip/wds \
  --output-prefix train --clip-len 512
```

输出目录位于 HDFS，不能直接替换为本地目录。打包器检查行数对齐，并使用重叠 50% 的滑动窗口；此设置下不超过 512 帧的序列不会产生训练片段。每个 tar 样本包含保存原始 float32 字节的 `.npy`、元数据 `.jsonl` 和 ID `.json`。训练读取器按 `embed_dim` 还原这些字节；tar 中该成员不是带文件头的标准 NumPy 文件。

## 4. 训练 Q-Clip

复制一份实验配置：

```bash
cp qclip/configs/dedup_tsa256.yaml qclip/configs/local.yaml
```

保留其中的 `model` 部分，添加以下 `trainer` 覆盖项，并将 `data` 部分替换为实际设置：

```yaml
trainer:
  devices: 1
  strategy: auto
data:
  batch_size: 8
  num_workers: 4
  train:
    params:
      root: /user/you/qclip/wds
      partition_cfg:
        part_size: 128
        interval_size: 256
  validation: null
  test: null
```

实验配置会与 `qclip/configs/default_config.yaml` 合并；后者默认使用八张 GPU 和内部 HDFS 路径。若保留 validation/test 数据集，也需替换对应路径。当前读取器固定使用 HDFS，本地训练需适配读取器。请根据 GPU 显存调整 batch size。

```bash
WANDB_MODE=offline python -m qclip.train \
  -c local --log-dir logs/qclip --exp my_experiment
```

`-c` 接受**不带 `.yaml` 后缀**的配置名。输出包括 `logs/qclip/my_experiment/` 下合并后的 `config.yaml` 和 Lightning checkpoint。在实验 YAML 中设置 `ckpt_path` 可恢复训练。提供的训练计划运行 20,000 步。

## 5. 编码片段

批量推理入口要求 `--ckpt-dir` 中包含 `config.yaml` 和 **`step=20000.ckpt`**。若采用其他训练步数，需调整加载器中的 checkpoint 路径，或直接使用 Python 加载函数。

```bash
python -m data.infer.infer_qclip \
  --task-jsonl /path/to/tasks.jsonl \
  --frame-emb-dir /path/to/frame_embeddings \
  --output-dir /path/to/clip_embeddings \
  --ckpt-dir logs/qclip/my_experiment \
  --clip-len 256 --batch-size 32
```

分别对数据库和查询视频运行，需 CUDA。输出 JSONL 每行形如 `{"entity_id":"123","clip_id":0,"slot_id":0}`。短序列直接聚合全部帧；长序列最后一个完整片段对齐到序列末尾，因此可能与前一个片段重叠。

## 6. 建库与评估

将数据库描述子及其元数据合并到**同一个**输出目录：

```bash
python -m tools.eval_merge_tuple \
  --input-emb-dir /path/to/database_clips \
  --input-info-dir /path/to/database_clips \
  --output-emb-dir /path/to/retrieval/index_merged \
  --output-info-dir /path/to/retrieval/index_merged --prefix index
```

对查询描述子重复上述步骤，将两个输出目录均设为 `query_merged`，并使用 `--prefix query`。然后检索：

```bash
python -m eval.faiss_retrieval --work-dir /path/to/retrieval --top-k 10
```

该入口归一化描述子，执行内积检索，并输出 `result/retrieval_results_no_pca.jsonl`。`--top-k` 表示视频级聚合前，**每个查询向量检索的向量数**。可用 `--pca-dim` 开启 PCA 白化。

准备 GT，每行是一条查询到目标的映射：

```json
{"query_id":"query_001","entity_id":"123"}
```

当前 `eval/cal_metrics.py` 的主入口运行硬编码示例，因此需显式调用 CLI 函数：

```bash
python -c "from eval.cal_metrics import main; main()" \
  --result-path /path/to/retrieval/result/retrieval_results_no_pca.jsonl \
  --gt-path /path/to/gt.jsonl \
  --query-id-key entity_id --target-id-key entity_id \
  --gt-query-key query_id --gt-label-key entity_id
```

`metrics.json` 和 `error_records.jsonl` 保存在检索结果旁。脚本按查询和目标 ID 累加得分；Recall@1/5/10 只统计聚合结果中出现、且存在 GT 的查询。没有保留检索结果的查询不计入分母，需另外核对查询覆盖率。若描述子 ID 对应单集视频，做剧集级评估前需将数据库 `entity_id` 映射到整部剧的 ID。上述脚本未实现 FIVR 的 Chamfer 相似度与 mAP 评估协议。

# Pipeline guide

[Back to README](../README.md) · **English** | [简体中文](REPRODUCING.zh-CN.md)

Run the commands below from the repository root in a Linux shell. Replace `/path/to/...` and HDFS paths with your own locations. Checkpoints and datasets are not bundled; prepare compatible frame-encoder and Q-Clip weights before inference.

## Environment

Use Python 3.11 and a CUDA-capable PyTorch environment. The full preprocessing and training stack targets Linux x86_64. In addition to the [core model environment](../README.md#getting-started), it needs:

- GPU FAISS compatible with your CUDA runtime. `requirements.txt` pins a Linux x86_64 CPython 3.11 wheel; it is not a portable installation recipe.
- CUDA toolkit, nvJPEG, a C++17 compiler, and CMake ≥3.18 to build the local `phash_cuda` extension.
- A working HDFS client (`hdfs dfs`), libhdfs, and the corresponding PyArrow/fsspec setup for training and WebDataset packing.
- ModelScope/TOS dependencies for the corresponding optional utilities. The full requirements include `bytedtos`.

Review those requirements before installing the full stack:

```bash
python -m pip install -r requirements.txt
# Direct dependencies used by the pipeline but not explicitly listed above:
python -m pip install omegaconf opencv-python pyarrow
```

The shell launchers in `scripts/` contain cluster-specific paths and environment variables. The module commands in this guide avoid those launchers.

## Data contracts

Keep video IDs unique and preserve frame order. Use the same file stem across frame metadata, frame embeddings, and clip embeddings:

| Artifact | Format |
| --- | --- |
| `meta/123.jsonl` | One retained frame per line, including `img_path` |
| `frame_embeddings/123.npy` | Float32 array `[num_frames, 1536]`, in metadata order |
| `clip_embeddings/123.npy` | Normalized array `[num_clips × num_slots, 1536]` |
| `clip_embeddings/123.jsonl` | One record per descriptor row |

Example frame metadata and task record (in separate JSONL files):

```json
{"video_id":"123","img_path":"/path/to/frames/123/frame_0000.jpg","frame_idx":0,"timestamp":0.0,"fps":25.0}
```

```json
{"jsonl_path":"/path/to/meta/123.jsonl"}
```

Task JSONL files drive feature extraction. The WebDataset packer instead takes a TXT file containing one metadata JSONL path per line. Its training IDs must be numeric: the reader converts `album_id` and `pair_id` to integers.

Do not L2-normalize frame features before aggregation unless your trained encoder/checkpoint expects it. Match the encoder, feature dimension, and checkpoint; the paper uses DINOv2 while this snapshot contains DINOv3 utilities.

## 1. Extract and filter frames

Prepare `videos.txt` with one local video path per line, then run:

```bash
python -m data.preprocess.process_video \
  --input /path/to/videos.txt \
  --output_frames_dir /path/to/frames \
  --output_meta_dir /path/to/meta \
  --target_fps 1
```

Quality filtering, pHash, and SSCD deduplication are enabled by default. SSCD expects `data/preprocess/modules/deduplicate/pretrained/sscd_disc_mixup.torchscript.pt`. `--no-sscd`, `--no-phash`, and `--no-entropy` disable individual stages; the imported modules still need their dependencies installed. This entry processes videos individually; corpus-wide deduplication requires grouping/processing the corpus separately.

## 2. Extract frame embeddings

`infer_dino_pretrain` reads `dinov3/pretrained/short_drama/config.yaml` and `teacher_checkpoint.pth`. Prepare those files with a compatible backbone; the paths are currently set in `build_default_model()`.

```bash
python -m data.infer.infer_dino_pretrain \
  --task-jsonl /path/to/tasks.jsonl \
  --output-dir /path/to/frame_embeddings \
  --batch-size 128 --num-workers 8
```

It writes `<metadata_stem>.npy`. Image-loading failures are skipped without rewriting the original metadata. Resolve failed frames and maintain exact metadata/embedding alignment before packing.

## 3. Pack training shards

```bash
python -m data.build_wds.build_wds_qclip \
  --task-txt /path/to/meta_paths.txt \
  --emb-dir /path/to/frame_embeddings \
  --output-dir /user/you/qclip/wds \
  --output-prefix train --clip-len 512
```

The output is HDFS, not a local directory. The packer checks row alignment and uses 50% overlapping windows; sequences of at most 512 frames produce no training clips with this setting. Each tar sample contains `.npy` raw float32 bytes, `.jsonl` metadata, and `.json` IDs. The training reader reshapes those raw bytes using `embed_dim`; this tar member is not a standard NumPy-file payload.

## 4. Train Q-Clip

Copy an experiment configuration:

```bash
cp qclip/configs/dedup_tsa256.yaml qclip/configs/local.yaml
```

Keep its `model` section. Add the following `trainer` overrides and replace its `data` section with your settings:

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

The experiment merges with `qclip/configs/default_config.yaml`, which otherwise selects eight GPUs and internal HDFS roots. If you retain validation/test datasets, replace their roots too. The current reader always uses HDFS; local training needs a reader adaptation. Adjust batch size to available GPU memory.

```bash
WANDB_MODE=offline python -m qclip.train \
  -c local --log-dir logs/qclip --exp my_experiment
```

`-c` takes the configuration name **without** `.yaml`. Outputs include the merged `config.yaml` and Lightning checkpoints under `logs/qclip/my_experiment/`. Set `ckpt_path` in the experiment YAML to resume. The provided schedule runs for 20,000 steps.

## 5. Encode clips

The batch inference entry expects `config.yaml` and **`step=20000.ckpt`** in `--ckpt-dir`. For other training lengths, adapt the checkpoint path in the loader or use its Python helper directly.

```bash
python -m data.infer.infer_qclip \
  --task-jsonl /path/to/tasks.jsonl \
  --frame-emb-dir /path/to/frame_embeddings \
  --output-dir /path/to/clip_embeddings \
  --ckpt-dir logs/qclip/my_experiment \
  --clip-len 256 --batch-size 32
```

Run this separately for database and query videos. CUDA is required. Each output JSONL record is `{"entity_id":"123","clip_id":0,"slot_id":0}`. Short sequences are aggregated as-is; the last full clip of longer sequences is aligned to the sequence end and can overlap the preceding clip.

## 6. Build the index and evaluate

Merge database descriptors and records into the **same** output directory:

```bash
python -m tools.eval_merge_tuple \
  --input-emb-dir /path/to/database_clips \
  --input-info-dir /path/to/database_clips \
  --output-emb-dir /path/to/retrieval/index_merged \
  --output-info-dir /path/to/retrieval/index_merged --prefix index
```

Repeat for query descriptors with `query_merged` as both output directories and `--prefix query`. Then search:

```bash
python -m eval.faiss_retrieval --work-dir /path/to/retrieval --top-k 10
```

This normalizes descriptors, performs inner-product search, and writes `result/retrieval_results_no_pca.jsonl`. `--top-k` is the number of retrieved **vectors per query vector**, before video-level aggregation. `--pca-dim` optionally enables PCA whitening.

Prepare ground truth, one query-to-target mapping per line:

```json
{"query_id":"query_001","entity_id":"123"}
```

Invoke the CLI function explicitly because this snapshot's `eval/cal_metrics.py` main block runs a hardcoded example:

```bash
python -c "from eval.cal_metrics import main; main()" \
  --result-path /path/to/retrieval/result/retrieval_results_no_pca.jsonl \
  --gt-path /path/to/gt.jsonl \
  --query-id-key entity_id --target-id-key entity_id \
  --gt-query-key query_id --gt-label-key entity_id
```

Outputs are `metrics.json` and `error_records.jsonl` beside the retrieval results. Scores are summed by query and target ID; Recall@1/5/10 uses only queries present in the aggregated results and GT. Queries with no retained results are excluded from the denominator, so verify query coverage separately. For series-level evaluation, map database `entity_id` values to series IDs before aggregation if your descriptors identify episodes. The scripts do not implement the FIVR Chamfer-similarity/mAP protocol.

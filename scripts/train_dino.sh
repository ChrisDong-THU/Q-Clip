git config --global --add safe.directory /mnt/bn/yuanliang-llm-gpu-train/jiajun/code/qclip
git config core.fileMode false

NNODES=${ARNOLD_WORKER_NUM:-1}
NODE_RANK=${ARNOLD_ID}
NPROC_PER_NODE=${ARNOLD_WORKER_GPU}
MASTER_ADDR=${ARNOLD_WORKER_0_HOST}
MASTER_PORT=${ARNOLD_WORKER_0_PORT//,/}

DATA_ROOT=/home/byte_toutiao_client_ai/ai_image_recognition_v3/wds

torchrun \
  --nnodes=${NNODES} \
  --node_rank=${NODE_RANK} \
  --nproc_per_node=${NPROC_PER_NODE}\
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} dinov3/train/train.py \
  --config-file dinov3/configs/train/dinov3_vitg16_pretrain.yaml \
  --output-dir logs/pretrain_vitg16_toutiao/train-2 \
  train.dataset_path=ToutiaoVideos:root=${DATA_ROOT}/20260212/movie,${DATA_ROOT}/20260305/movie,${DATA_ROOT}/20260305/tv_series:resample=True:shardshuffle=1000

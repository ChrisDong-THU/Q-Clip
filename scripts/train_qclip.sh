git config --global --add safe.directory /mnt/bn/yuanliang-llm-gpu-train/jiajun/code/qclip
git config core.fileMode false

export MASTER_ADDR=${ARNOLD_WORKER_0_HOST}
export MASTER_PORT=${ARNOLD_WORKER_0_PORT//,/}
export NODE_RANK=0

export WANDB_MODE=offline
export PROJECT=qclip

CFG=$1
EXP=$2

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python qclip/train.py -c $CFG --exp $EXP
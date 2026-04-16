source keep_gpu_alive.sh && start_gpu_alive

NODE_ID=$1
SPLIT=$2

python data/infer/infer_dino_pretrain_tos.py \
    --task-jsonl tmp_tos_tasks_0413/${SPLIT}/task_node_${NODE_ID}.jsonl \
    --output-dir tmp_infer_dino_emb/${SPLIT}

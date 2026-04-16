source keep_gpu_alive.sh && start_gpu_alive

NODE_ID=$1
SPLIT=$2

python data/infer/infer_dino_pretrain.py \
    --task-jsonl tmp_tasks_hdfs_0113/${SPLIT}/task_node_${NODE_ID}_bbox.jsonl \
    --output-dir tmp_infer_dino_emb/${SPLIT}

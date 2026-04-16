source keep_gpu_alive.sh && start_gpu_alive

NODE_ID=$1

python data/aux/cal_bbox.py \
    --task-txt tmp_tasks_hdfs_0113/short_drama_dedup/task_node_${NODE_ID}.txt \
    --sample-ratio 0.125
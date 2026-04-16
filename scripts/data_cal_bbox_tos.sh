source keep_gpu_alive.sh && start_gpu_alive

NODE_ID=$1
SPLIT=$2

python data/aux/cal_bbox_tos.py \
    --task-txt tmp_tos_tasks_0413/${SPLIT}/task_node_${NODE_ID}.txt 
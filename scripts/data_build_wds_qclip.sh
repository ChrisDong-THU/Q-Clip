source keep_gpu_alive.sh && start_gpu_alive

NODE_ID=$1
SPLIT=$2

python data/build_wds/build_wds_qclip.py \
    --task-txt tmp_wds_tasks_0413/${SPLIT}/task_node_${NODE_ID}.txt \
    --emb-dir tmp_infer_dino_emb/${SPLIT}/emb \
    --output-dir hdfs://haruna/home/byte_toutiao_client_ai/ai_image_recognition_v3/wds_qclip/20260415/${SPLIT} \
    --output-prefix ${NODE_ID}
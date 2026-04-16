source keep_gpu_alive.sh && start_gpu_alive

pip install data/preprocess/modules/deduplicate/phash_cuda

NODE_ID=$1
OUTPUT_DIR=./tmp_meta_k400/train

python data/preprocess/process_video.py \
    --input tmp_k400_tasks/train/task_node_${NODE_ID}.txt \
    --output_frames_dir ${OUTPUT_DIR}/frames \
    --output_meta_dir ${OUTPUT_DIR}/meta \
    --target_fps 2 \
    --no-entropy \
    --no-phash \
    --no-sscd \
    --upload-dir frames/20260414/k400/train

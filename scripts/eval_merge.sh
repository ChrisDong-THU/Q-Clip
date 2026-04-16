SPLIT=short_drama
OUTPUT_DIR=tmp_faiss_dino_uni

python tools/eval_merge_tuple.py \
    --input-emb-dir tmp_infer_dino_emb/${SPLIT}/emb \
    --input-info-dir tmp_infer_dino_emb/${SPLIT}/emb_info \
    --output-emb-dir ${OUTPUT_DIR}/index_full_merged \
    --output-info-dir ${OUTPUT_DIR}/index_full_merged \
    --prefix index

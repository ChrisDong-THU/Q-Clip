python eval/cal_metrics.py \
    --result-path tmp_faiss_dino_dedup/result/retrieval_results_no_pca.jsonl \
    --gt-path tmp_bench_short_drama/data/short_drama_v2.jsonl \
    --query-id-key gid \
    --target-id-key album_id \
    --gt-query-key gid \
    --gt-label-key entity_id \
    --min-score 0.0
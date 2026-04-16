"""检索评估工具：根据检索结果和 GT 计算 Recall 指标。"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


# ============== Utils ==============

def load_jsonl(path: str | Path) -> list[dict]:
    """加载 JSONL 文件。"""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading"):
            if line.strip():
                records.append(json.loads(line))
    
    return records


def load_gt_map(gt_path: str | Path, query_key: str, label_key: str) -> dict[str, str]:
    """加载 GT 映射：query_entity_id -> label_entity_id。"""
    gt_map = {}
    for record in load_jsonl(gt_path):
        gt_map[str(record[query_key])] = str(record[label_key])
    return gt_map


# ============== Evaluation ==============

def aggregate_scores(
    results: list[dict],
    query_id_key: str,
    target_id_key: str,
    min_score: float = 0.0,
) -> dict[str, dict[str, float]]:
    """聚合检索结果：过滤低于 min_score 的结果，同一 query 对同一 target 取最高分后累加。"""
    entity_scores: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    filtered_count = 0
    for r in tqdm(results, desc="Aggregating"):
        score = r["score"]
        if score < min_score:
            filtered_count += 1
            continue

        q_id = str(r["q_record"][query_id_key])
        t_id = str(r["tgt_record"][target_id_key])
        
        entity_scores[q_id][t_id] += score

    if filtered_count > 0:
        print(f"[INFO] Filtered {filtered_count} results below min_score={min_score}")

    return entity_scores


def compute_recall(
    entity_scores: dict[str, dict[str, float]],
    gt_map: dict[str, str],
    top_k_list: list[int] = [1, 5, 10],
) -> tuple[dict[str, float], list[dict]]:
    """计算 Recall@K 指标。"""
    recalls = {k: 0 for k in top_k_list}
    error_records = []

    for q_id, target_scores in entity_scores.items():
        gt_id = gt_map.get(q_id)
        if gt_id is None:
            continue

        sorted_targets = sorted(target_scores.items(), key=lambda x: -x[1])
        top_ids = [t_id for t_id, _ in sorted_targets]

        hit = False
        for k in top_k_list:
            if gt_id in top_ids[:k]:
                recalls[k] += 1
                hit = True

        if not hit:
            error_records.append({
                "query_id": q_id,
                "gt_id": gt_id,
                "pred": [{"id": t_id, "score": round(s, 4)} for t_id, s in sorted_targets[:5]],
            })

    total = len([q for q in entity_scores if q in gt_map])
    recall_rates = {f"Recall@{k}": recalls[k] / total if total > 0 else 0 for k in top_k_list}

    return recall_rates, error_records


# ============== Main ==============

def run_evaluation(
    result_path: str,
    gt_path: str,
    query_id_key: str = "gid",
    target_id_key: str = "album_id",
    gt_query_key: str = "gid",
    gt_label_key: str = "entity_id",
    min_score: float = 0.0,
) -> None:
    """执行检索评估。"""
    output_dir = Path(result_path).parent

    # 加载数据
    print(f"[INFO] Loading results from {result_path}")
    results = load_jsonl(result_path)
    print(f"[INFO] Loaded {len(results)} results")

    print(f"[INFO] Loading GT from {gt_path}")
    gt_map = load_gt_map(gt_path, gt_query_key, gt_label_key)
    print(f"[INFO] Loaded {len(gt_map)} GT mappings")

    # 聚合分数（在此处过滤 min_score）
    entity_scores = aggregate_scores(results, query_id_key, target_id_key, min_score)
    print(f"[INFO] Aggregated {len(entity_scores)} query entities")

    # 计算指标
    recall_rates, error_records = compute_recall(entity_scores, gt_map)

    # 输出结果
    print("\n" + "=" * 40)
    for metric, value in recall_rates.items():
        print(f"{metric}: {value:.4f}")
    print("=" * 40)

    # 保存错误记录
    error_path = output_dir / "error_records.jsonl"
    with open(error_path, "w", encoding="utf-8") as f:
        for record in error_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\n[DONE] Saved {len(error_records)} error records to {error_path}")

    # 保存指标
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(recall_rates, f, indent=2)
    print(f"[DONE] Saved metrics to {metrics_path}")


def main():
    parser = argparse.ArgumentParser(
        description="检索评估工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--result-path", required=True, help="检索结果文件路径")
    parser.add_argument("--gt-path", required=True, help="GT 文件路径")
    parser.add_argument("--query-id-key", default="gid", help="查询记录中的 ID 字段名")
    parser.add_argument("--target-id-key", default="album_id", help="目标记录中的 ID 字段名")
    parser.add_argument("--gt-query-key", default="gid", help="GT 中的查询 ID 字段名")
    parser.add_argument("--gt-label-key", default="entity_id", help="GT 中的标签 ID 字段名")
    parser.add_argument("--min-score", type=float, default=0.0, help="最小分数阈值")
    args = parser.parse_args()

    run_evaluation(
        args.result_path,
        args.gt_path,
        args.query_id_key,
        args.target_id_key,
        args.gt_query_key,
        args.gt_label_key,
        args.min_score,
    )


if __name__ == "__main__":
    # main()

    run_evaluation(
        "tmp_faiss_dino_uni/result/retrieval_results_no_pca.jsonl",
        "tmp_bench_short_drama/data/short_drama_v2.jsonl",
        "gid",
        "album_id",
        "gid",
        "entity_id",
        0.0,
    )
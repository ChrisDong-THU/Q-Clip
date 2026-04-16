"""TOS 目录列举工具：列举指定前缀下的所有子目录并按修改时间过滤。"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import bytedtos
from tqdm import tqdm


# ============== TOS Client ==============

class TosClient:
    """TOS 对象存储客户端。"""

    def __init__(
        self,
        bucket_name: str,
        ak: str,
        tos_psm: str,
        tos_cluster: str,
        endpoint: str,
        delimiter: str = "/",
        max_keys: int = 1000,
        timeout: int = 30,
    ):
        self.client = bytedtos.Client(
            bucket_name, ak,
            service=tos_psm, cluster=tos_cluster, endpoint=endpoint,
            timeout=timeout, connect_timeout=timeout,
        )
        self.delimiter = delimiter
        self.max_keys = max_keys

    def list_dirs(self, prefix: str, start_after: str = "") -> tuple[list[str], str, bool]:
        """列举子目录（单页），返回 (目录列表, 下次游标, 是否截断)。"""
        try:
            resp = self.client.list_prefix(prefix, self.delimiter, start_after, self.max_keys)
            data = resp.json["payload"]
            return data["commonPrefix"], data["startAfter"], data["isTruncated"]
        except Exception as e:
            print(f"[ERROR] List dirs failed: {prefix}, {e}")
            return [], "", False

    def list_all_dirs(self, prefix: str) -> list[str]:
        """列举所有子目录（自动分页）。"""
        all_dirs, start_after, is_truncated = [], "", True
        while is_truncated:
            dirs, start_after, is_truncated = self.list_dirs(prefix, start_after)
            all_dirs.extend(dirs)
        return all_dirs

    def get_dir_last_modified(self, prefix: str) -> Optional[int]:
        """获取目录内第一个对象的修改时间戳。"""
        try:
            resp = self.client.list_prefix(prefix, self.delimiter, "", 1)
            objects = resp.json["payload"]["objects"]
            if not objects:
                return None

            last_modified = objects[0].get("lastModified")
            dt = datetime.strptime(last_modified, "%a, %d %b %Y %H:%M:%S %Z")
            return int(dt.replace(tzinfo=timezone.utc).timestamp())
        except Exception:
            return None


# ============== Utils ==============

def parse_date(date_str: str) -> int:
    """解析日期字符串为时间戳。"""
    return int(datetime.strptime(date_str, "%Y-%m-%d").timestamp())


# ============== Main ==============

def list_tos_dirs(
    tos_config: dict,
    prefix: str,
    output_txt: str,
    after_date: Optional[str] = None,
) -> int:
    """
    列举 TOS 目录并按修改时间过滤，输出到文件。
    
    Args:
        tos_config: TOS 客户端配置
        prefix: TOS 目录前缀
        output_txt: 输出文件路径
        after_date: 仅保留此日期之后修改的目录 (格式: YYYY-MM-DD)
    
    Returns:
        输出的目录数量
    """
    tos_client = TosClient(**tos_config)
    after_ts = parse_date(after_date) if after_date else None

    # 确保输出目录存在
    Path(output_txt).parent.mkdir(parents=True, exist_ok=True)

    # 列举并过滤
    filtered_dirs = []
    start_after, is_truncated = "", True
    pbar = tqdm(desc="Listing dirs")

    while is_truncated:
        dirs, start_after, is_truncated = tos_client.list_dirs(prefix, start_after)

        for d in dirs:
            pbar.update(1)

            if after_ts:
                last_modified = tos_client.get_dir_last_modified(d)
                if last_modified is None or last_modified < after_ts:
                    continue

            filtered_dirs.append(d)

    pbar.close()

    # 写入输出文件
    with open(output_txt, "w", encoding="utf-8") as f:
        for d in filtered_dirs:
            f.write(d + "\n")

    print(f"[DONE] {len(filtered_dirs)} dirs written to {output_txt}")
    if after_date:
        print(f"       (filtered by modified after {after_date})")

    return len(filtered_dirs)


if __name__ == "__main__":
    tos_config = {
        "ak": "8X4STUMNH9EIFO5FFHE1",
        "bucket_name": "toutiao-aigc-video-frames",
        "tos_psm": "toutiao.tos.tosapi",
        "tos_cluster": "default",
        "endpoint": "tos-cn-north.byted.org",
        "delimiter": "/",
        "max_keys": 100,
    }

    entity_type = "playlet"

    list_tos_dirs(tos_config, f"ai_image_recognition_v3_origin/{entity_type}/", f"tmp_tos_tasks_0413/{entity_type}/task_all.txt", "2026-03-01")


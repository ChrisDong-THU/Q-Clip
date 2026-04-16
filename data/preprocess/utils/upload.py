from typing import List
import subprocess
import os
import concurrent.futures


HDFS_ROOT = "hdfs://haruna/home/byte_toutiao_client_ai/ai_image_recognition_v3/"


def get_hdfs_dir(local_dir: str) -> str:
    """
    Get the HDFS directory path for a local directory
    """
    return os.path.join(HDFS_ROOT, local_dir)


def upload_files_to_hdfs(local_files: str | List[str], hdfs_dir: str, max_workers: int = 10) -> None:
    """
    Upload a single file or multiple files from local to HDFS directory in parallel
    """
    if isinstance(local_files, str):
        local_files = [local_files]
    
    # Check if hdfs_dir starts with HDFS_ROOT
    if not hdfs_dir.startswith(HDFS_ROOT):
        raise ValueError(f"hdfs_dir {hdfs_dir} must start with {HDFS_ROOT}")
    
    subprocess.run([
        'hdfs', 'dfs', '-mkdir', '-p', hdfs_dir
    ], check=True)
    
    def upload_worker(local_file: str) -> None:
        filename = os.path.basename(local_file)
        hdfs_path = os.path.join(hdfs_dir, filename)

        subprocess.run([
            'hdfs', 'dfs', '-put', local_file, hdfs_path
        ], check=True)

    # Upload files in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(upload_worker, local_files)
    
    print(f"[INFO] Successfully uploaded {len(local_files)} files to {hdfs_dir}")
    

def upload_dir_to_hdfs(local_dir: str, hdfs_dir: str) -> None:
    """
    Upload a local directory to HDFS directory
    """
    if not os.path.isdir(local_dir):
        raise ValueError(f"{local_dir} is not a directory")
    
    # Check if hdfs_dir starts with HDFS_ROOT
    if not hdfs_dir.startswith(HDFS_ROOT):
        raise ValueError(f"hdfs_dir {hdfs_dir} must start with {HDFS_ROOT}")
    
    subprocess.run([
        'hdfs', 'dfs', '-mkdir', '-p', hdfs_dir
    ], check=True)
    
    subprocess.run([
        'hdfs', 'dfs', '-put', local_dir, hdfs_dir
    ], check=True)
    
    print(f"[INFO] Successfully uploaded {local_dir} to {hdfs_dir}")
    
    
def prune_img_paths(img_paths: List[str], keep_img_paths: List[str]) -> None:
    """
    Prune a list of image paths, remove files not in keep_img_paths
    """
    keep_img_paths = set(keep_img_paths)
    
    for img_path in img_paths:
        if img_path not in keep_img_paths:
            os.remove(img_path)
    
    print(f"[INFO] Successfully pruned {len(img_paths)} files, keep {len(keep_img_paths)} files")
    
    
if __name__ == "__main__":
    upload_files_to_hdfs(["test1.txt", "test2.txt"], "tmp")

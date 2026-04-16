import os
import time
import random
import requests
import subprocess
import uuid
import shutil
from urllib.parse import urlparse


def is_av1(video_path):
    try:
        out = subprocess.check_output(
            f'ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of csv=p=0 "{video_path}"',
            shell=True
        ).decode().strip()
        return out == "av1"
    except:
        return False


def transcode(video_path):
    tmp = video_path + ".tmp.mp4"
    subprocess.run(
        f'ffmpeg -threads 4 -y -i "{video_path}" -c:v libx264 -crf 23 -preset fast -c:a copy "{tmp}"',
        shell=True,
        check=True
    )
    os.replace(tmp, video_path)


def download_video_by_gid(gid, output_dir, timeout=300, max_retries: int = 3, backoff_cap_s: int = 60) -> str:
    file_name = f"{gid}.mp4"
    output_path = os.path.join(output_dir, file_name)

    if os.path.exists(output_path):
        print(f"[INFO] Video {gid} already exists at {output_path}, skipping download.")
        return output_path

    url = "https://woo9wet4.fn.bytedance.net/admin/video/gid_to_url"
    payload = {"gid": gid}

    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()

            video_url = data.get("play_url")  # 这是 M3U8 地址
            video_path = urlparse(video_url).path.lower()
            
            if video_path.endswith("m3u8"):
                download_m3u8_by_N_m3u8DL_RE(video_url, file_name, output_dir)
            else:
                download_video_by_http(video_url, file_name, output_dir)

            if is_av1(output_path):
                print(f"[INFO] Video {gid} is AV1, transcoding to H264.")
                transcode(output_path)
            
            return output_path
            
        except Exception as e:
            if attempt == max_retries - 1:
                raise Exception(f"Failed to download video {gid} after {max_retries} retries. Error: {e}")
            
            backoff_s = min(backoff_cap_s, 2**attempt) + random.random()
            print(f"[INFO] Retry {attempt + 1} for {gid} after {backoff_s:.2f} seconds. Error: {e}")
            time.sleep(backoff_s)


def download_video_by_http(url, file_name, output_dir):
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()

    with open(os.path.join(output_dir, file_name), 'wb') as f:
        f.write(resp.content)
    
    return True


def download_m3u8_by_N_m3u8DL_RE(url, file_name, output_dir, threads_num=8) -> bool:
    unique_id = uuid.uuid4().hex[:8]
    temp_dir = os.path.join(output_dir, f"tmp_{file_name}_{unique_id}")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)
    command = [
        'N_m3u8DL-RE', url,
        '--concurrent-download', 'true',
        '--save-dir', output_dir,
        '--tmp-dir', temp_dir,
        '--save-name', file_name,
        '--thread-count', str(threads_num),
        '--download-retry-count', '3',
        '--no-log',
        '--log-level', 'ERROR'
    ]
    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
        )
        return True
    except Exception as e:
        print(f"[WARN] Failed to download M3U8 {url}: Error: {e}")
        return False
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_gid = 7526418555441597746
    download_video_by_gid(test_gid, output_dir="./tmp_video_fake")
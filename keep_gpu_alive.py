import torch
import os
import time
import multiprocessing


def keep_gpu_busy(gpu_id):
    size = int((2 * 1024 * 1024) ** 0.5)  # 4 bytes per float32
    a = torch.randn(size, size, device=gpu_id)
    b = torch.randn(size, size, device=gpu_id)
    
    while True:
        c = torch.matmul(a, b)
        time.sleep(1e-3)


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
num_gpus = torch.cuda.device_count()


processes = []
for i in range(num_gpus):
    p = multiprocessing.Process(target=keep_gpu_busy, args=(i,))
    p.start()
    processes.append(p)

for p in processes:
    p.join()
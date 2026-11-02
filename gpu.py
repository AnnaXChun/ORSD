# import torch

# import os
# import time
# import argparse


# def gpu_info(gpu_id=0):
#    gpu_status = os.popen('nvidia-smi -i {} | grep %'.format(gpu_id)).read().split('|')
#    # | N/A   34C    P0    72W / 400W |  69193MiB / 81251MiB |      0%      Default |
#    gpu_memory = int(gpu_status[2].split('/')[0].split('M')[0].strip())
#    gpu_power = int(gpu_status[1].split('   ')[-1].split('/')[0].split('W')[0].strip())
#    gpu_utilization = int(gpu_status[3].split('%')[0].strip())
#    return gpu_power, gpu_memory, gpu_utilization


# if __name__ == '__main__':
#    parser = argparse.ArgumentParser()
#    parser.add_argument(
#        "--gpu_number",
#        required=False,
#        default="8",
#        type=int,
#    )
#    args = parser.parse_args()

#    inputs1 = []
#    inputs2 = []
#    for i in range(args.gpu_number): #['0', '1', '2', '3', '4', '5', '6', '7']:
#        device = torch.device('cuda:'+str(i))
#        inputs1.append(torch.rand(1000, 500, dtype=torch.float).to(device))
#        inputs2.append(torch.rand(500, 1000, dtype=torch.float).to(device))

#    check_time = time.time()
#    not_busy_devices = range(args.gpu_number)
#    while True:
#        now_time = time.time()
#        if (now_time - check_time) > 50:
#            not_busy_devices = []
#            for i in range(args.gpu_number):
#                gpu_power, gpu_memory, gpu_utilization = gpu_info(i)
#                if gpu_utilization < 50:
#                    not_busy_devices.append(i)
#            check_time = time.time()
#        for i in not_busy_devices:
#            f = inputs1[i] @ inputs2[i]

import torch
import time
import subprocess
import multiprocessing as mp

# === 可调参数 ===
THRESHOLD = 50          # 率阈值 (%)
CHECK_INTERVAL = 10    # 每次检测间隔 (秒)
RUN_SECONDS = 50         # run时间 (秒)

def get_gpu_utilization():
    """返回每张GPU的利用率列表"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
        utils = [int(x.strip()) for x in result.stdout.strip().split("\n")]
        return utils
    except Exception as e:
        print(f"[Error] 无法获取GPU利用率: {e}")
        return []

def gpu_task(gpu_id, tensor_size_mb=1, run_seconds=RUN_SECONDS):
    """单GPU占用任务，持续运行 run_seconds 秒"""
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(device)

    elem_count = tensor_size_mb * 1024 * 1024 // 2
    matrix_side = int(elem_count ** 0.62)
    data_a = torch.randn(matrix_side, matrix_side, dtype=torch.float16, device=device)
    data_b = torch.randn(matrix_side, matrix_side, dtype=torch.float16, device=device)

    end_time = time.time() + run_seconds
    while time.time() < end_time:
        result = torch.matmul(data_a, data_b)
        data_a = result.detach()
        torch.cuda.synchronize()

    del data_a, data_b
    torch.cuda.empty_cache()

def monitor_and_run(threshold=THRESHOLD, check_interval=CHECK_INTERVAL, run_seconds=RUN_SECONDS):
    """监控 GPU 利用率，小于 threshold 时占用 GPU"""

    while True:
        utils = get_gpu_utilization()
        if not utils:
            time.sleep(check_interval)
            continue

        print("当前GPU利用率:", utils)

        low_util_gpus = [i for i, u in enumerate(utils) if u < threshold]
        if low_util_gpus:
            procs = [mp.Process(target=gpu_task, args=(i, 1, run_seconds)) for i in low_util_gpus]
            for p in procs:
                p.start()
            for p in procs:
                p.join()

        time.sleep(check_interval)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    monitor_and_run()
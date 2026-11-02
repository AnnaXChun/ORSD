"""Opt-in process-level CUDA diagnostics; never frees memory or changes tensors."""

import json
import os
import socket
import subprocess
import time
from pathlib import Path


def memory_audit(stage, **details):
    if os.environ.get("VERL_MEMORY_AUDIT", "0") != "1":
        return None
    record = dict(stage=stage, time=time.time(), host=socket.gethostname(),
                  pid=os.getpid(), ppid=os.getppid(),
                  visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), **details)
    try:
        record["namespace_pids"] = next(
            (line.split()[1:] for line in Path("/proc/self/status").read_text().splitlines()
             if line.startswith("NSpid:")), [])
    except OSError:
        pass
    try:
        import torch
        # Do not create a CUDA context in a controller-only process.
        if torch.cuda.is_initialized():
            device = torch.cuda.current_device()
            torch.cuda.synchronize(device)
            free, total = torch.cuda.mem_get_info(device)
            record.update(device=device, gpu_uuid=str(getattr(
                torch.cuda.get_device_properties(device), "uuid", "unavailable")),
                free_bytes=free, total_bytes=total,
                allocated_bytes=torch.cuda.memory_allocated(device),
                reserved_bytes=torch.cuda.memory_reserved(device),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
        result = subprocess.run([
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        record["gpu_processes"] = result.stdout.strip().splitlines()
        record["nvidia_smi_returncode"] = result.returncode
    except Exception as exc:
        record["diagnostic_error"] = f"{type(exc).__name__}: {exc}"
    print("[memory-audit] " + json.dumps(record, sort_keys=True), flush=True)
    return record

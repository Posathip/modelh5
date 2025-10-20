#!/usr/bin/env python3
# ============================================================
# ✅ Inference Benchmark Script (No Plots) with CPU Usage
# ============================================================

import numpy as np
import tensorflow as tf
import psutil
import os
import platform
import subprocess
import time
import json
from tensorflow.keras.models import load_model
import threading

# ============================================================
# 🔹 Environment Information
# ============================================================
def get_gpu_info():
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
             '--format=csv,noheader,nounits'],
            encoding='utf-8'
        ).strip().split('\n')
        return [r.split(', ') for r in result]
    except Exception:
        return None

print("\n🧠 Environment Info")
print(f"System     : {platform.system()} {platform.release()}")
print(f"Processor  : {platform.processor()}")
print(f"CPU Cores  : {psutil.cpu_count(logical=True)}")
print(f"RAM (GB)   : {psutil.virtual_memory().total / (1024**3):.2f}")

gpu_info = get_gpu_info()
if gpu_info:
    for idx, g in enumerate(gpu_info):
        print(f"GPU {idx}: {g[0]}, {g[1]} MB VRAM, Driver {g[2]}")
else:
    print("GPU        : None detected (CPU mode)")

# ============================================================
# 🔹 Load Model
# ============================================================
model_path = "cnn_bilstm_multihead_model.keras"
print(f"\n📦 Loading model from {model_path} ...")
model = tf.keras.models.load_model(model_path)
print("✅ Model loaded successfully!")

# ============================================================
# 🔹 Prepare Dummy Test Data (same shape as training)
# ============================================================
input_dim = model.input_shape[1]
X_test = np.random.rand(1000, input_dim, 1).astype(np.float32)  # 1,000 samples

# ============================================================
# 🔹 Run Inference Benchmark with CPU usage tracking
# ============================================================
cpu_usage_list = []
stop_tracking = [False]

def track_cpu_usage():
    while not stop_tracking[0]:
        cpu_usage_list.append(psutil.cpu_percent(interval=0.1))

print("\n⚡ Running inference benchmark ...")
t = threading.Thread(target=track_cpu_usage)
t.start()

start = time.time()
_ = model.predict(X_test, verbose=0)
end = time.time()

stop_tracking[0] = True
t.join()

total_time = end - start
avg_time_per_sample = (total_time / len(X_test)) * 1000  # ms/sample
avg_cpu = sum(cpu_usage_list) / len(cpu_usage_list)

print(f"\n✅ Inference complete.")
print(f"Total inference time: {total_time:.4f} sec")
print(f"Average per sample  : {avg_time_per_sample:.4f} ms/sample")
print(f"Average CPU usage   : {avg_cpu:.2f} %")

# ============================================================
# 🔹 Save Log File
# ============================================================
log = {
    "System": platform.system(),
    "Processor": platform.processor(),
    "CPU_Cores": psutil.cpu_count(logical=True),
    "RAM_GB": round(psutil.virtual_memory().total / (1024**3), 2),
    "GPU": gpu_info[0][0] if gpu_info else "None",
    "Total_Inference_s": round(total_time, 4),
    "Avg_ms_per_sample": round(avg_time_per_sample, 4),
    "Avg_CPU_percent": round(avg_cpu, 2)
}

with open("inference_log.json", "w") as f:
    json.dump(log, f, indent=4)

print("\n📝 Log saved as inference_log.json")

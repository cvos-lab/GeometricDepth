# utils/io_utils.py
import numpy as np
import subprocess

def get_gpu_temperature():
    """Returns GPU temperature in Celsius, or None if unavailable."""
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits']
        )
        return int(result.decode().strip())
    except Exception:
        return None

def read_Z(path, img_height, img_width):
    with open(path, 'r') as f:
        data = np.array(f.read().split())
    reshaped = data.reshape(img_height, img_width)
    return reshaped.astype(float)

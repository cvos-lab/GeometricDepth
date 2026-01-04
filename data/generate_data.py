import os
import glob
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from utils.io_utils import read_Z
import random

IMG_H, IMG_W = 512, 768
NUM_SAMPLES = 3070
FRAME_INDEX = 8
BG_FRAME_INDEX = 28
BG_SAMPLE_INDEX = 2306

TEST_INDICES = [
    0, 17, 79, 85, 187, 259, 301, 1045, 1190, 1192, 1196, 1198, 1201, 1205,
    1208, 1217, 1229, 1234, 1253, 1269, 1467, 1471, 1472, 1619, 1629, 1620,
    1801, 1806, 1833, 2311, 3043, 3048, 3, 49, 76, 122, 130, 146, 152, 168,
    202, 214, 235, 255, 271, 300, 366, 379, 405, 477, 502, 550, 599, 640,
    679, 701, 749, 800, 823, 853, 901, 937, 1012, 1089, 1152, 1204, 1257, 1300,
    1350, 1450, 1502, 1554, 1600, 1650
]

Z_MIN = -200.0 * 3.1954
Z_MAX = 150.0 * 3.1954


def generate_and_save_pt():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    save_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data_pt")
    os.makedirs(save_dir, exist_ok=True)

    Z_paths = sorted(glob.glob(os.path.join(root_dir, "depth_dataset", "left_cam", "depth_labels", "*.txt")))
    fringe_paths = sorted(glob.glob(os.path.join(root_dir, "depth_dataset", "left_cam", "images", "*.png")))
    frames_per_sample = 30

    assert len(Z_paths) == NUM_SAMPLES, f"Expected {NUM_SAMPLES} Z files, found {len(Z_paths)}"
    assert len(fringe_paths) >= NUM_SAMPLES * frames_per_sample, \
        f"Expected {NUM_SAMPLES * frames_per_sample} fringe images, found {len(fringe_paths)}"

    # --- PROCESS REFERENCE FRINGE (Sample 2306, Frame 8) ---
    print(f"📥 Processing Reference Fringe from Sample {BG_SAMPLE_INDEX} Frame {FRAME_INDEX}...")
    ref_f_path = fringe_paths[BG_SAMPLE_INDEX * frames_per_sample + FRAME_INDEX]
    ref_f_raw = np.array(Image.open(ref_f_path)).astype(np.float32)
    norm_ref_f = ref_f_raw / 127.5 - 1.0
    ref_f_tensor = torch.from_numpy(norm_ref_f).float().unsqueeze(0).unsqueeze(0)
    torch.save(ref_f_tensor, os.path.join(save_dir, "ref_fringe.pt"))
    print(f"✅ Saved ref_fringe.pt")

    # --- PROCESS REFERENCE BACKGROUND (Sample 2306, Frame 28) ---
    print(f"📥 Processing Reference BG from Sample {BG_SAMPLE_INDEX} Frame {BG_FRAME_INDEX}...")
    ref_bg_path = fringe_paths[BG_SAMPLE_INDEX * frames_per_sample + BG_FRAME_INDEX]
    ref_bg_raw = np.array(Image.open(ref_bg_path)).astype(np.float32)
    norm_ref_bg = ref_bg_raw / 127.5 - 1.0
    ref_bg_tensor = torch.from_numpy(norm_ref_bg).float().unsqueeze(0).unsqueeze(0)
    torch.save(ref_bg_tensor, os.path.join(save_dir, "ref_bg.pt"))
    print(f"✅ Saved ref_bg.pt")
    # -----------------------------------------------------------

    fringe_tensor = torch.zeros((NUM_SAMPLES, 1, IMG_H, IMG_W), dtype=torch.float32)
    Z_tensor = torch.zeros((NUM_SAMPLES, 1, IMG_H, IMG_W), dtype=torch.float32)

    print("📥 Loading raw images and depth maps...")
    for i in tqdm(range(NUM_SAMPLES), desc="Generating dataset"):
        Z = read_Z(Z_paths[i], IMG_H, IMG_W)
        Z[Z == -1000.0] = Z_MIN

        Z_normalized = (Z - Z_MIN) / (Z_MAX - Z_MIN)
        Z_tensor[i, 0] = torch.from_numpy(Z_normalized).float()

        frame_path = fringe_paths[i * frames_per_sample + FRAME_INDEX]
        fringe = np.array(Image.open(frame_path)).astype(np.float32)
        fringe[Z == Z_MIN] = 0
        norm_fringe = fringe / 127.5 - 1.0
        fringe_tensor[i, 0] = torch.from_numpy(norm_fringe).float()

    all_indices = set(range(NUM_SAMPLES))
    test_indices = sorted(set(TEST_INDICES))
    trainval_indices = sorted(all_indices - set(test_indices))

    fringe_trainval = fringe_tensor[trainval_indices]
    Z_trainval = Z_tensor[trainval_indices]
    fringe_test = fringe_tensor[test_indices]
    Z_test = Z_tensor[test_indices]

    torch.save(fringe_trainval, os.path.join(save_dir, "fringe.pt"))
    torch.save(Z_trainval, os.path.join(save_dir, "Z.pt"))

    torch.save({
        "data": fringe_test,
        "original_indices": test_indices
    }, os.path.join(save_dir, "fringe_test.pt"))

    torch.save({
        "data": Z_test,
        "original_indices": test_indices
    }, os.path.join(save_dir, "Z_test.pt"))

    print(f"✅ Saved to: {save_dir}")
    print(f"   - {len(trainval_indices)} samples to trainval")
    print(f"   - {len(test_indices)} samples to test")


if __name__ == "__main__":
    generate_and_save_pt()
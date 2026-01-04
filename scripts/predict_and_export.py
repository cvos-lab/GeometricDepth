import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from data.dataset import PrecomputedFringeDepthDataset
from models.HiDNet import HiDNet
from collections import OrderedDict
import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---- HARDCODED SETTINGS (Must match Training) ----
MODEL_OUT_CHANNELS = 1
MODEL_BASE_FILTERS = 32
MODEL_DROPOUT = 0.2
MODEL_ALPHA = 0.3

IMG_H, IMG_W = 512, 768
Z_MIN = -200.0 * 3.1954
Z_MAX = 150.0 * 3.1954

CAMERA_INTRINSICS = [
    2.7674498064097711e+03,
    2.7675590254576309e+03,
    3.8258601201085355e+02,
    2.0311869177718791e+02
]


def save_3D(filepath, data):
    with open(filepath, 'w') as f:
        for i in range(IMG_W):
            for j in range(IMG_H):
                f.write(f"{i} {IMG_H - 1 - j} {data[j, i]:.6f}\n")


def load_model_checkpoint(model, checkpoint_path, device='cpu'):
    state_dict = torch.load(checkpoint_path, map_location=device)
    if any(k.startswith('module.') for k in state_dict.keys()):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace('module.', '')] = v
        state_dict = new_state_dict
    model.load_state_dict(state_dict)
    return model


def save_input_images(loader, output_dir, original_indices):
    for mapped_idx, batch in enumerate(tqdm(loader, desc="Saving Input Images (.png)")):
        original_idx = original_indices[mapped_idx]

        fringe_tensor = batch["fringe"]
        fringe_tensor = fringe_tensor[:, 0:1, :, :]

        norm_fringe = fringe_tensor.squeeze().cpu().numpy()
        fringe_vis = (norm_fringe + 1.0) * 127.5
        fringe_vis = np.clip(fringe_vis, 0, 255).astype(np.uint8)
        plt.imsave(os.path.join(output_dir, f"Input_{original_idx:04d}.png"), fringe_vis, cmap='gray')


def predict_3D(model, loader, device, output_dir, original_indices):
    model.eval()
    for mapped_idx, batch in enumerate(tqdm(loader, desc="Predicting (3D .txt)")):
        original_idx = original_indices[mapped_idx]
        fringe = batch["fringe"].to(device)
        label = batch["Z"].to(device)
        with torch.no_grad():
            output = model(fringe)
            Z_pred = output
        Z_pred_np = Z_pred.squeeze().cpu().numpy() * (Z_MAX - Z_MIN) + Z_MIN
        Z_gt_np = label.squeeze().cpu().numpy() * (Z_MAX - Z_MIN) + Z_MIN
        save_3D(os.path.join(output_dir, f"3D_{original_idx:04d}_pred.txt"), Z_pred_np)
        save_3D(os.path.join(output_dir, f"3D_{original_idx:04d}_gt.txt"), Z_gt_np)

def run_prediction():
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(project_dir, "data", "data_pt")
    output_dir = os.path.join(project_dir, "output", "results")
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("   🎛️  SELECT INPUT MODE")
    print("=" * 60)
    print(" [1] I only (1 Ch)")
    print(" [2] I + RefFringe + Diff (3 Ch)")
    print(" [3] I + U + V (3 Ch)")
    print(" [4] I + RefFringe + Diff + U + V (5 Ch)")
    print(" [5] I + RefFringe + Diff + RefBG (4 Ch)")
    print(" [6] I + RefFringe + Diff + U + V + RefBG (6 Ch)")
    print("=" * 60)

    choice = input("👉 Enter your choice (1-6): ").strip()

    mapping = {
        "1": (1, 1, False, False),
        "2": (3, 2, True, False),
        "3": (3, 3, False, False),
        "4": (5, 4, True, False),
        "5": (4, 5, True, True),
        "6": (6, 6, True, True)
    }

    if choice in mapping:
        in_channels, selected_option, need_ref_f, need_ref_bg = mapping[choice]

        ref_fringe_path = os.path.join(data_dir, "ref_fringe.pt") if need_ref_f else None
        if need_ref_f and not os.path.exists(ref_fringe_path):
            ref_fringe_path = os.path.join(data_dir, "bg.pt")

        ref_bg_path = os.path.join(data_dir, "ref_bg.pt") if need_ref_bg else None

        print(f"\n✅ Selected Option {selected_option} (In Channels: {in_channels})\n")
    else:
        print("❌ Invalid input.")
        return

    fringe_data = torch.load(os.path.join(data_dir, "fringe_test.pt"))
    original_indices = fringe_data["original_indices"]
    print("✅ Available test indices:")
    print(original_indices)

    dataset = PrecomputedFringeDepthDataset(
        fringe_path=os.path.join(data_dir, "fringe_test.pt"),
        Z_path=os.path.join(data_dir, "Z_test.pt"),
        ref_fringe_path=ref_fringe_path,
        ref_bg_path=ref_bg_path,
        option=selected_option,
        intrinsics=CAMERA_INTRINSICS
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HiDNet(
        in_channels=in_channels,
        out_channels=MODEL_OUT_CHANNELS,
        base_filters=MODEL_BASE_FILTERS,
        dropout=MODEL_DROPOUT,
        alpha=MODEL_ALPHA
    ).to(device)

    ckpt_path = os.path.join(project_dir, "checkpoints", "hidnet_best.pth")
    if not os.path.exists(ckpt_path):
        print(f"⚠️ Checkpoint not found at {ckpt_path}")
        return

    model = load_model_checkpoint(model, ckpt_path, device=device)

    save_input_images(loader, output_dir, original_indices)
    predict_3D(model, loader, device, output_dir, original_indices)

if __name__ == "__main__":
    run_prediction()
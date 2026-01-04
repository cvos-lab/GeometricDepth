import os

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

import yaml
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.io_utils import get_gpu_temperature
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt

import random
import numpy as np

from data.dataset import PrecomputedFringeDepthDataset
from models.HiDNet import HiDNet

import matplotlib

matplotlib.use('Agg')

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

CAMERA_INTRINSICS = [
    2.7674498064097711e+03,
    2.7675590254576309e+03,
    3.8258601201085355e+02,
    2.0311869177718791e+02
]


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_config(path="configs/default_hidnet.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_model(name, config):
    if name.lower() == "hidnet":
        return HiDNet(
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            base_filters=config["base_filters"],
            dropout=config["dropout"],
            alpha=config["alpha"]
        )
    else:
        raise ValueError(f"Unknown model name: {name}")


def compute_loss(pred, target, criterion, ssim_loss, config):
    if config["loss_function"] == "mse":
        return criterion(pred, target)
    elif config["loss_function"] == "ssim":
        return 1 - ssim_loss(pred, target)
    elif config["loss_function"] == "combined":
        mse_loss = criterion(pred, target)
        ssim_value = ssim_loss(pred, target)
        return config["loss_alpha"] * mse_loss + config["beta"] * (1 - ssim_value)
    else:
        raise ValueError(f"Invalid loss function: {config['loss_function']}")


def train_model(model, config, model_name="hidnet", selected_option=1):
    print("=" * 45)
    print(f"🚀 STARTING TRAINING for {model_name.upper()} | Option {selected_option}")
    print("=" * 45)

    start_time = time.time()

    project_dir = os.path.abspath(os.path.dirname(__file__))
    checkpoint_dir = os.path.join(project_dir, config["checkpoint_dir"])
    os.makedirs(checkpoint_dir, exist_ok=True)
    data_dir = os.path.join(project_dir, config["data_dir"])

    use_ref_fringe = selected_option in [2, 4, 5, 6]
    ref_fringe_path = os.path.join(data_dir, "ref_fringe.pt")
    if use_ref_fringe and not os.path.exists(ref_fringe_path):
        ref_fringe_path = os.path.join(data_dir, "bg.pt")
    if not use_ref_fringe:
        ref_fringe_path = None

    use_ref_bg = selected_option in [5, 6]
    ref_bg_path = os.path.join(data_dir, "ref_bg.pt") if use_ref_bg else None

    dataset = PrecomputedFringeDepthDataset(
        fringe_path=os.path.join(data_dir, "fringe.pt"),
        Z_path=os.path.join(data_dir, "Z.pt"),
        ref_fringe_path=ref_fringe_path,
        ref_bg_path=ref_bg_path,
        option=selected_option,
        intrinsics=CAMERA_INTRINSICS
    )

    val_split = config["validation_split"]
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=config["shuffle"],
        worker_init_fn=seed_worker,
        num_workers=4,
        generator=generator,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        worker_init_fn=seed_worker,
        num_workers=4,
        generator=generator,
        pin_memory=True,
    )

    if torch.cuda.device_count() > 1:
        print(f"🖥️ Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    T_max = config["lr_schedule"]["T_max"]
    eta_min = config["lr_schedule"]["eta_min"]
    scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    criterion = nn.MSELoss()
    ssim_loss = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()

    best_val_loss = float("inf")
    best_epoch = -1
    train_losses = []
    val_losses = []

    try:
        for epoch in range(T_max):
            model.train()
            train_loss = 0.0

            print(f"\nEpoch {epoch + 1}/{T_max}")
            progress_bar = tqdm(train_loader, desc="Training", ncols=100)

            for batch in progress_bar:
                x = batch["fringe"].cuda()
                y = batch["Z"].cuda()

                optimizer.zero_grad()
                outputs = model(x)
                loss = compute_loss(outputs, y, criterion, ssim_loss, config)
                final_output = outputs
                loss.backward()
                optimizer.step()

                loss_for_logging = criterion(final_output, y)
                train_loss += loss_for_logging.item() * x.size(0)
                avg_batch_loss = loss_for_logging.item()
                progress_bar.set_postfix(loss=f"{avg_batch_loss:.6f}")

            gpu_temp = get_gpu_temperature()
            print(f"🌡️ GPU Temp: {gpu_temp}°C")

            avg_train_loss = train_loss / train_size
            train_losses.append(avg_train_loss)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["fringe"].cuda()
                    y = batch["Z"].cuda()
                    outputs = model(x)
                    loss = criterion(outputs, y)
                    val_loss += loss.item() * x.size(0)

            avg_val_loss = val_loss / val_size
            val_losses.append(avg_val_loss)

            print(f"Epoch [{epoch + 1}/{T_max}] - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"{model_name}_best.pth"))

            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f"Adjusted LR at epoch {epoch + 1}: {current_lr:.10f}")

    except KeyboardInterrupt:
        print("\n⛔ Training interrupted by user. Saving current model...")
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"{model_name}_interrupted.pth"))
        print("✅ Model saved. Exiting.")
        return

    final_lr = scheduler.get_last_lr()[0]
    print(f"\n📉 Final LR: {final_lr:.10f}")
    print(f"🔚 Final Train Loss: {train_losses[-1]:.6f}")
    print(f"🔍 Final Val Loss: {val_losses[-1]:.6f}")

    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name.upper()} Loss Curve")
    plt.legend()
    plt.grid(True)

    ymin, ymax = plt.gca().get_ylim()
    y_offset = (ymax - ymin) * 0.05

    if best_epoch > 0:
        plt.annotate(
            f"{best_val_loss:.6f}",
            xy=(best_epoch, best_val_loss),
            xytext=(best_epoch, best_val_loss + y_offset),
            arrowprops=dict(arrowstyle="->", color="red", lw=1),
            fontsize=12,
            color="red",
            weight="bold",
            ha="center"
        )

    plt.savefig(os.path.join(checkpoint_dir, f"{model_name}_loss.png"))
    plt.close()

    print(f"\n✅ Finished training {model_name.upper()} - Best Val Loss: {best_val_loss:.6f} (Epoch {best_epoch})\n")
    duration = time.time() - start_time
    print(f"⏱️ Total training time for {model_name.upper()}: {duration / 60:.2f} minutes")


def main():
    config = load_config()

    print("\n" + "=" * 50)
    print("   🎛️  SELECT INPUT MODE")
    print("=" * 50)
    print(" [1] I only (1 Ch)")
    print(" [2] I + RefFringe + Diff (3 Ch)")
    print(" [3] I + U + V (3 Ch)")
    print(" [4] I + RefFringe + Diff + U + V (5 Ch)")
    print(" [5] I + RefFringe + Diff + RefBG (4 Ch)")
    print(" [6] I + RefFringe + Diff + U + V + RefBG (6 Ch)")
    print("=" * 50)

    selected_option = 1

    while True:
        choice = input("👉 Enter your choice (1-6): ").strip()
        mapping = {
            "1": (1, 1), "2": (3, 2), "3": (3, 3),
            "4": (5, 4), "5": (4, 5), "6": (6, 6)
        }

        if choice in mapping:
            config["in_channels"], selected_option = mapping[choice]
            print(f"\n✅ Selected Option {selected_option} (In Channels: {config['in_channels']})\n")
            break
        else:
            print("❌ Invalid input. Please type a number between 1 and 6.")

    model = get_model("hidnet", config)
    train_model(model, config, model_name="hidnet", selected_option=selected_option)


if __name__ == "__main__":
    main()
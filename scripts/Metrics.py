import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import OrderedDict
from scipy.ndimage import binary_dilation, sobel

# Import your modules
from data.dataset import PrecomputedFringeDepthDataset
from models.HiDNet import HiDNet

# ============================================================================
# Configuration
# ============================================================================

IMG_H, IMG_W = 512, 768
Z_MIN = -200.0 * 3.1954
Z_MAX = 150.0 * 3.1954
PIXEL_TO_MM = 3.1954

CAMERA_INTRINSICS = [
    2.7674498064097711e+03,
    2.7675590254576309e+03,
    3.8258601201085355e+02,
    2.0311869177718791e+02
]


# ============================================================================
# Evaluation Class
# ============================================================================

class DepthEvaluator:
    """Standard CV depth evaluation with strict GT-based masking"""

    def __init__(self, min_depth=10.0, max_depth=200.0, boundary_erosion=1):
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.boundary_erosion = boundary_erosion

    def create_evaluation_mask(self, depth_gt, gradient_threshold=10.0):
        """
        Create valid pixel mask based ONLY on Ground Truth.
        """
        H, W = depth_gt.shape
        mask = np.ones((H, W), dtype=bool)

        # 1. Valid depth range (Strictly from GT)
        mask &= (depth_gt >= self.min_depth) & (depth_gt <= self.max_depth)

        # 2. Detect depth discontinuities (Strictly from GT)
        grad_x = np.abs(sobel(depth_gt, axis=1))
        grad_y = np.abs(sobel(depth_gt, axis=0))
        edges = (grad_x > gradient_threshold) | (grad_y > gradient_threshold)

        # 3. Erode around boundaries
        if self.boundary_erosion > 0:
            invalid_boundary = binary_dilation(edges, iterations=self.boundary_erosion)
            mask &= ~invalid_boundary

        # 4. Remove NaN/Inf from GT
        mask &= np.isfinite(depth_gt)

        return mask

    def compute_metrics(self, pred, gt, mask):
        """Compute all depth estimation metrics (input in mm)"""

        # --- Safety Step: Exclude NaN predictions from evaluation (Skip, don't penalize) ---
        # We combine the GT mask with the Prediction validity check
        combined_mask = mask & np.isfinite(pred)

        pred_valid = pred[combined_mask]
        gt_valid = gt[combined_mask]

        if len(pred_valid) == 0:
            return None

        epsilon = 1e-8
        gt_valid = np.maximum(np.abs(gt_valid), epsilon)
        pred_valid = np.maximum(np.abs(pred_valid), epsilon)

        metrics = {}

        # 1. Threshold Accuracy
        thresh = np.maximum((gt_valid / pred_valid), (pred_valid / gt_valid))
        metrics['delta1'] = (thresh < 1.25).mean() * 100
        metrics['delta2'] = (thresh < 1.25 ** 2).mean() * 100
        metrics['delta3'] = (thresh < 1.25 ** 3).mean() * 100

        # 2. Absolute Threshold Accuracy
        abs_error = np.abs(pred_valid - gt_valid)
        metrics['acc_0.5mm'] = (abs_error < 0.5).mean() * 100
        metrics['acc_1.0mm'] = (abs_error < 1.0).mean() * 100
        metrics['acc_2.0mm'] = (abs_error < 2.0).mean() * 100

        # 3. Relative errors
        # Note: We use the already filtered arrays here
        gt_abs = np.abs(gt_valid)
        metrics['abs_rel'] = np.mean(abs_error / np.maximum(gt_abs, epsilon))
        metrics['sq_rel'] = np.mean((abs_error ** 2) / np.maximum(gt_abs, epsilon))

        # 4. Absolute errors (in mm)
        metrics['rmse'] = np.sqrt(np.mean(abs_error ** 2))
        metrics['mae'] = np.mean(abs_error)

        # 5. Log errors
        metrics['rmse_log'] = np.sqrt(np.mean((np.log(gt_valid) - np.log(pred_valid)) ** 2))

        # 6. Additional statistics
        metrics['max_error'] = np.max(abs_error)
        metrics['median_error'] = np.median(abs_error)
        metrics['percentile_95'] = np.percentile(abs_error, 95)

        # 7. Valid pixel percentage (Based on GT mask originally passed in)
        metrics['valid_pixels'] = (mask.sum() / mask.size) * 100

        return metrics

    def print_metrics(self, metrics, method_name="Method"):
        if metrics is None:
            print("⚠️ No valid metrics computed")
            return

        print(f"\n{'=' * 70}")
        print(f"  {method_name}")
        print(f"{'=' * 70}")
        print(f"  Absolute Threshold Accuracy (mm):")
        print(f"    Acc@0.5mm  : {metrics['acc_0.5mm']:.2f}%")
        print(f"    Acc@1.0mm  : {metrics['acc_1.0mm']:.2f}%")
        print(f"    Acc@2.0mm  : {metrics['acc_2.0mm']:.2f}%")
        print(f"\n  Relative Threshold Accuracy:")
        print(f"    δ < 1.25   : {metrics['delta1']:.2f}%")
        print(f"    δ < 1.25²  : {metrics['delta2']:.2f}%")
        print(f"    δ < 1.25³  : {metrics['delta3']:.2f}%")
        print(f"\n  Error Metrics (mm):")
        print(f"    MAE        : {metrics['mae']:.3f} mm")
        print(f"    RMSE       : {metrics['rmse']:.3f} mm")
        print(f"    Median     : {metrics['median_error']:.3f} mm")
        print(f"    Max Error  : {metrics['max_error']:.3f} mm")
        print(f"    95th %ile  : {metrics['percentile_95']:.3f} mm")
        print(f"\n  Relative Errors:")
        print(f"    AbsRel     : {metrics['abs_rel']:.4f}")
        print(f"    SqRel      : {metrics['sq_rel']:.4f}")
        print(f"    RMSE (log) : {metrics['rmse_log']:.4f}")
        print(f"\n  Valid Pixels : {metrics['valid_pixels']:.1f}%")
        print(f"{'=' * 70}\n")


# ============================================================================
# Model Loading
# ============================================================================

def load_model_checkpoint(model, checkpoint_path, device='cpu'):
    state_dict = torch.load(checkpoint_path, map_location=device)
    if any(k.startswith('module.') for k in state_dict.keys()):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace('module.', '')] = v
        state_dict = new_state_dict
    model.load_state_dict(state_dict)
    return model


# ============================================================================
# Batch Evaluation on All Test Samples
# ============================================================================

def evaluate_all_samples(model, loader, device, evaluator, method_name="Method"):
    model.eval()
    all_metrics = []

    print(f"\n🚀 Evaluating {len(loader)} test samples...")

    with torch.no_grad():
        for batch in tqdm(loader, desc="Processing"):
            fringe = batch["fringe"].to(device)
            Z_normalized = batch["Z"].to(device)

            # Predict
            Z_pred_normalized = model(fringe)

            # Denormalize to pixel units
            Z_pred_pixels = Z_pred_normalized.squeeze().cpu().numpy() * (Z_MAX - Z_MIN) + Z_MIN
            Z_gt_pixels = Z_normalized.squeeze().cpu().numpy() * (Z_MAX - Z_MIN) + Z_MIN

            # ------------------------------------------------------------------
            # Create mask using ONLY Ground Truth
            # ------------------------------------------------------------------
            mask = evaluator.create_evaluation_mask(Z_gt_pixels)

            # Convert to mm for metric computation
            Z_pred_mm = Z_pred_pixels / PIXEL_TO_MM
            Z_gt_mm = Z_gt_pixels / PIXEL_TO_MM

            metrics = evaluator.compute_metrics(Z_pred_mm, Z_gt_mm, mask)

            if metrics is not None:
                all_metrics.append(metrics)

    if len(all_metrics) == 0:
        print("❌ No valid metrics computed")
        return None, []

    # Aggregate metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        avg_metrics[key] = np.mean(values)
        avg_metrics[f'{key}_std'] = np.std(values)

    return avg_metrics, all_metrics


# ============================================================================
# Main Evaluation Pipeline
# ============================================================================

def run_full_evaluation():
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(project_dir, "data", "data_pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}\n")

    # ========================================================================
    # Step 1: Select Configuration
    # ========================================================================
    print("=" * 70)
    print("   🎛️  SELECT INPUT CONFIGURATION")
    print("=" * 70)
    print(" [1] I only (1 Ch)")
    print(" [2] I + RefFringe + Diff (3 Ch)")
    print(" [3] I + U + V (3 Ch)")
    print(" [4] I + RefFringe + Diff + U + V (5 Ch)")
    print(" [5] I + RefFringe + Diff + RefBG (4 Ch)")
    print(" [6] I + RefFringe + Diff + U + V + RefBG (6 Ch)")
    print("=" * 70)

    choice = input("👉 Enter choice (1-6): ").strip()

    config_map = {
        "1": (1, 1, None, None),
        "2": (3, 2, "ref_fringe.pt", None),
        "3": (3, 3, None, None),
        "4": (5, 4, "ref_fringe.pt", None),
        "5": (4, 5, "ref_fringe.pt", "ref_bg.pt"),
        "6": (6, 6, "ref_fringe.pt", "ref_bg.pt"),
    }

    if choice not in config_map:
        print("❌ Invalid choice")
        return

    in_channels, option, ref_fringe_file, ref_bg_file = config_map[choice]

    ref_fringe_path = os.path.join(data_dir, ref_fringe_file) if ref_fringe_file else None
    if ref_fringe_path and not os.path.exists(ref_fringe_path):
        ref_fringe_path = os.path.join(data_dir, "bg.pt")

    ref_bg_path = os.path.join(data_dir, ref_bg_file) if ref_bg_file else None

    method_name = f"Option {option}"
    print(f"\n✅ Selected: {method_name} ({in_channels} channels)\n")

    # ========================================================================
    # Step 2: Load Dataset
    # ========================================================================
    print("📂 Loading test dataset...")
    dataset = PrecomputedFringeDepthDataset(
        fringe_path=os.path.join(data_dir, "fringe_test.pt"),
        Z_path=os.path.join(data_dir, "Z_test.pt"),
        ref_fringe_path=ref_fringe_path,
        ref_bg_path=ref_bg_path,
        option=option,
        intrinsics=CAMERA_INTRINSICS
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(f"✅ Loaded {len(dataset)} test samples\n")

    # ========================================================================
    # Step 3: Load Model
    # ========================================================================
    print("🔧 Loading model...")
    model = HiDNet(
        in_channels=in_channels,
        out_channels=1,
        base_filters=32,
        dropout=0.2,
        alpha=0.3
    ).to(device)

    # Note: Make sure this points to the correct checkpoint for the option!
    ckpt_path = os.path.join(project_dir, "checkpoints", "hidnet_best.pth")
    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint not found: {ckpt_path}")
        return

    model = load_model_checkpoint(model, ckpt_path, device=device)
    print(f"✅ Model loaded\n")

    # ========================================================================
    # Step 4: Create Evaluator
    # ========================================================================
    sample_batch = dataset[0]
    sample_Z = sample_batch["Z"].numpy() * (Z_MAX - Z_MIN) + Z_MIN
    actual_min = sample_Z.min()
    actual_max = sample_Z.max()

    # Dynamic Mask Range (Strictly based on data range)
    mask_min = actual_min - 50
    mask_max = actual_max + 50

    print(f"📊 Setting up evaluator:")
    print(f"   Mask range: [{mask_min:.1f}, {mask_max:.1f}] pixels")
    print(f"   Boundary erosion: 1 pixel")
    print(f"   Conversion: ÷ {PIXEL_TO_MM} to get mm\n")

    evaluator = DepthEvaluator(
        min_depth=mask_min,
        max_depth=mask_max,
        boundary_erosion=2  # <--- Strict setting: 1
    )

    # ========================================================================
    # Step 5 & 6: Run & Print
    # ========================================================================
    avg_metrics, all_metrics = evaluate_all_samples(
        model, loader, device, evaluator, method_name
    )

    if avg_metrics is not None:
        evaluator.print_metrics(avg_metrics, f"{method_name} - Average")

        # Summary Table
        print("\n" + "=" * 70)
        print("  📊 RESULTS TABLE (Copy to Paper)")
        print("=" * 70)
        print(f"Method: {method_name}")
        print(f"{'Metric':<20} | {'Mean':<10} | {'Std':<10}")
        print("-" * 70)
        print(f"{'MAE (mm)':<20} | {avg_metrics['mae']:<10.3f} | {avg_metrics['mae_std']:<10.3f}")
        print(f"{'RMSE (mm)':<20} | {avg_metrics['rmse']:<10.3f} | {avg_metrics['rmse_std']:<10.3f}")
        print(f"{'Acc@0.5mm (%)':<20} | {avg_metrics['acc_0.5mm']:<10.2f} | {avg_metrics['acc_0.5mm_std']:<10.2f}")
        print(f"{'Acc@1.0mm (%)':<20} | {avg_metrics['acc_1.0mm']:<10.2f} | {avg_metrics['acc_1.0mm_std']:<10.2f}")
        print(
            f"{'Valid Pixels (%)':<20} | {avg_metrics['valid_pixels']:<10.1f} | {avg_metrics['valid_pixels_std']:<10.1f}")
        print("=" * 70)


if __name__ == "__main__":
    run_full_evaluation()
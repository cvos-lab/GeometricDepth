import torch
import matplotlib
matplotlib.use("TkAgg")  # or 'QtAgg' if needed
import matplotlib.pyplot as plt

def visualize_sample_batch(fringe_trainval_path, Z_trainval_path,
                           fringe_test_path, Z_test_path):
    """Visualize 10 key samples (3+3+4) from trainval and test sets."""

    # Load data
    trainval_fringe = torch.load(fringe_trainval_path)
    trainval_Z = torch.load(Z_trainval_path)

    test_data = torch.load(fringe_test_path)
    test_fringe = test_data["data"]
    test_indices = test_data["original_indices"]

    Z_test_data = torch.load(Z_test_path)
    test_Z = Z_test_data["data"]

    def get_indices(total):
        idxs = list(range(3))  # first 3
        mid = total // 2
        idxs += [mid - 1, mid, mid + 1]  # mid 3
        idxs += list(range(total - 4, total))  # last 4
        return [i for i in idxs if i < total]

    train_indices = get_indices(trainval_fringe.shape[0])
    test_local_indices = get_indices(test_fringe.shape[0])

    for i in range(len(train_indices)):
        fig, axs = plt.subplots(2, 2, figsize=(10, 6))
        fig.suptitle(f"Sample Pair {i+1} / {len(train_indices)}", fontsize=14)

        # Trainval
        idx = train_indices[i]
        axs[0, 0].imshow(trainval_fringe[idx, 0].numpy(), cmap='gray')
        axs[0, 0].set_title(f"Train Fringe [{idx}]")
        axs[0, 0].axis("off")

        axs[0, 1].imshow(trainval_Z[idx, 0].numpy(), cmap='inferno')
        axs[0, 1].set_title(f"Train Z [{idx}]")
        axs[0, 1].axis("off")

        # Test
        if i < len(test_local_indices):
            test_idx = test_local_indices[i]
            original_idx = test_indices[test_idx]

            axs[1, 0].imshow(test_fringe[test_idx, 0].numpy(), cmap='gray')
            axs[1, 0].set_title(f"Test Fringe [{original_idx}]")
            axs[1, 0].axis("off")

            axs[1, 1].imshow(test_Z[test_idx, 0].numpy(), cmap='inferno')
            axs[1, 1].set_title(f"Test Z [{original_idx}]")
            axs[1, 1].axis("off")
        else:
            axs[1, 0].axis("off")
            axs[1, 1].axis("off")

        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        plt.show()

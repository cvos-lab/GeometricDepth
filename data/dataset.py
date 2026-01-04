import torch
import numpy as np
from torch.utils.data import Dataset


class PrecomputedFringeDepthDataset(Dataset):
    def __init__(self, fringe_path, Z_path, ref_fringe_path=None, ref_bg_path=None, option=1, intrinsics=None):
        self.option = option

        self.fringe = torch.load(fringe_path)
        if isinstance(self.fringe, dict):
            self.original_indices = self.fringe["original_indices"]
            self.fringe = self.fringe["data"]
        else:
            self.original_indices = None

        self.Z = torch.load(Z_path)
        if isinstance(self.Z, dict):
            self.Z = self.Z["data"]

        self.ref_fringe = None
        if ref_fringe_path is not None:
            raw_ref = torch.load(ref_fringe_path)
            if isinstance(raw_ref, dict):
                self.ref_fringe = raw_ref["data"]
            else:
                self.ref_fringe = raw_ref

        self.ref_bg = None
        if ref_bg_path is not None:
            raw_bg = torch.load(ref_bg_path)
            if isinstance(raw_bg, dict):
                self.ref_bg = raw_bg["data"]
            else:
                self.ref_bg = raw_bg

        self.u_tensor = None
        self.v_tensor = None

        if self.fringe.ndim == 4:
            _, _, H, W = self.fringe.shape
        else:
            _, H, W = self.fringe.shape

        if self.option in [3, 4, 6]:
            if intrinsics is None:
                raise ValueError("Intrinsics required for Options 3, 4, 6")
            fx, fy, cx, cy = intrinsics
            us = np.arange(W, dtype=np.float32)
            vs = np.arange(H, dtype=np.float32)
            u_grid = (us - cx) / fx
            v_grid = (vs - cy) / fy
            u_map = np.tile(u_grid[None, :], (H, 1))
            v_map = np.tile(v_grid[:, None], (1, W))
            self.u_tensor = torch.from_numpy(u_map).unsqueeze(0)
            self.v_tensor = torch.from_numpy(v_map).unsqueeze(0)

    def __len__(self):
        return self.fringe.shape[0]

    def __getitem__(self, idx):
        I, target_Z = self.fringe[idx], self.Z[idx]

        if self.option == 1:
            return {"fringe": I, "Z": target_Z}

        elif self.option == 3:
            combined_input = torch.cat([I, self.u_tensor, self.v_tensor], dim=0)
            return {"fringe": combined_input, "Z": target_Z}

        else:
            if self.ref_fringe is None:
                raise ValueError(f"Option {self.option} requires ref_fringe path")

            REF_F = self.ref_fringe[0]
            Diff = (I - REF_F) / 2.0

            if self.option == 2:
                combined_input = torch.cat([I, REF_F, Diff], dim=0)

            elif self.option == 4:
                combined_input = torch.cat([I, REF_F, Diff, self.u_tensor, self.v_tensor], dim=0)

            elif self.option == 5:
                if self.ref_bg is None:
                    raise ValueError("Option 5 requires ref_bg path")
                REF_BG = self.ref_bg[0]
                combined_input = torch.cat([I, REF_F, Diff, REF_BG], dim=0)

            elif self.option == 6:
                if self.ref_bg is None:
                    raise ValueError("Option 6 requires ref_bg path")
                REF_BG = self.ref_bg[0]
                combined_input = torch.cat([I, REF_F, Diff, self.u_tensor, self.v_tensor, REF_BG], dim=0)

            else:
                raise ValueError("Invalid option selected")

            return {"fringe": combined_input, "Z": target_Z}

    def get_original_index(self, idx):
        return self.original_indices[idx] if self.original_indices is not None else idx
"""
Unified Data Loaders for All Four Datasets

Supports:
- Convection-Diffusion (1D)
- Traffic Flow (1D)
- Shallow Water (2D)
- Spinodal Decomposition (2D)

Training modes:
- onestep: returns (input_t, target_{t+ndt})
- pushforward: returns (input_t, [target_{t+ndt}, target_{t+2*ndt}, ..., target_{t+K*ndt}])
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.multiprocessing import get_context
from pathlib import Path
from typing import Tuple, List, Optional


class ConvectionDiffusionDataset(Dataset):
    """
    1D Convection-Diffusion Dataset

    Fields:
        - c: concentration [time, length] - conserved, evolves over time
        - u: convection velocity [length] - external field, constant in time

    Input: [c_t, u] (2 channels)
    Target (onestep): c_{t+ndt} (1 channel)
    Target (pushforward): [c_{t+ndt}, c_{t+2*ndt}, ..., c_{t+K*ndt}] (K channels)
    """

    def __init__(self, folder_path: str, ndt: int = 1, training_mode: str = 'onestep', unroll_steps: int = 5):
        self.folder_path = folder_path
        self.ndt = ndt
        self.training_mode = training_mode
        self.unroll_steps = unroll_steps
        self.data_info = []

        # Compute required future steps
        if training_mode == 'pushforward':
            required_future = ndt * unroll_steps
        else:
            required_future = ndt

        # Scan all h5 files
        for file_path in Path(folder_path).glob("*.h5"):
            with h5py.File(file_path, 'r') as f:
                n_timesteps = f['c'].shape[0]
                n_samples = n_timesteps - required_future

                for i in range(n_samples):
                    self.data_info.append({
                        'file': str(file_path),
                        'index': i
                    })

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        info = self.data_info[idx]

        with h5py.File(info['file'], 'r') as f:
            # Get current concentration
            c_current = f['c'][info['index']].astype(np.float32)

            # Get external velocity field (constant)
            u = f['u'][:].astype(np.float32)

            if self.training_mode == 'pushforward':
                # Get multiple future states
                targets = []
                for k in range(1, self.unroll_steps + 1):
                    c_future = f['c'][info['index'] + k * self.ndt].astype(np.float32)
                    targets.append(c_future[np.newaxis, :])
                target_tensor = np.concatenate(targets, axis=0)  # [K, length]
            else:
                # Single step
                c_next = f['c'][info['index'] + self.ndt].astype(np.float32)
                target_tensor = c_next[np.newaxis, :]  # [1, length]

        # Stack as input: [c, u]
        input_tensor = np.stack([c_current, u], axis=0)  # [2, length]

        return torch.from_numpy(input_tensor), torch.from_numpy(target_tensor)


class TrafficFlowDataset(Dataset):
    """
    1D Traffic Flow Dataset

    Fields:
        - rho: car density [time, length] - conserved, [0, 1], evolves
        - vmax: max velocity [length] - external field, constant in time

    Input: [rho_t, vmax] (2 channels)
    Target: rho_{t+ndt} or [rho_{t+ndt}, ..., rho_{t+K*ndt}]
    """

    def __init__(self, folder_path: str, ndt: int = 1, training_mode: str = 'onestep', unroll_steps: int = 5):
        self.folder_path = folder_path
        self.ndt = ndt
        self.training_mode = training_mode
        self.unroll_steps = unroll_steps
        self.data_info = []

        if training_mode == 'pushforward':
            required_future = ndt * unroll_steps
        else:
            required_future = ndt

        for file_path in Path(folder_path).glob("*.h5"):
            with h5py.File(file_path, 'r') as f:
                n_timesteps = f['rho'].shape[0]
                n_samples = n_timesteps - required_future

                for i in range(n_samples):
                    self.data_info.append({
                        'file': str(file_path),
                        'index': i
                    })

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        info = self.data_info[idx]

        with h5py.File(info['file'], 'r') as f:
            rho_current = f['rho'][info['index']].astype(np.float32)
            vmax = f['vmax'][:].astype(np.float32)

            if self.training_mode == 'pushforward':
                targets = []
                for k in range(1, self.unroll_steps + 1):
                    rho_future = f['rho'][info['index'] + k * self.ndt].astype(np.float32)
                    targets.append(rho_future[np.newaxis, :])
                target_tensor = np.concatenate(targets, axis=0)
            else:
                rho_next = f['rho'][info['index'] + self.ndt].astype(np.float32)
                target_tensor = rho_next[np.newaxis, :]

        input_tensor = np.stack([rho_current, vmax], axis=0)

        return torch.from_numpy(input_tensor), torch.from_numpy(target_tensor)


class ShallowWaterDataset(Dataset):
    """
    2D Shallow Water Dataset

    Fields:
        - h: water depth [time, H, W] - conserved, >= 0
        - mx: x-momentum [time, H, W] - conserved, unbounded
        - my: y-momentum [time, H, W] - conserved, unbounded

    Input: [h_t, mx_t, my_t] (3 channels)
    Target: [h_{t+ndt}, mx_{t+ndt}, my_{t+ndt}] or multiple steps
    """

    def __init__(self, folder_path: str, ndt: int = 1, training_mode: str = 'onestep', unroll_steps: int = 5):
        self.folder_path = folder_path
        self.ndt = ndt
        self.training_mode = training_mode
        self.unroll_steps = unroll_steps
        self.data_info = []

        if training_mode == 'pushforward':
            required_future = ndt * unroll_steps
        else:
            required_future = ndt

        for file_path in Path(folder_path).glob("*.h5"):
            with h5py.File(file_path, 'r') as f:
                n_timesteps = f['h'].shape[0]
                n_samples = n_timesteps - required_future

                for i in range(n_samples):
                    self.data_info.append({
                        'file': str(file_path),
                        'index': i
                    })

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        info = self.data_info[idx]

        with h5py.File(info['file'], 'r') as f:
            # Current state
            h_current = f['h'][info['index']].astype(np.float32)
            mx_current = f['mx'][info['index']].astype(np.float32)
            my_current = f['my'][info['index']].astype(np.float32)

            if self.training_mode == 'pushforward':
                # For pushforward, return list of future states
                # Shape: [K, 3, H, W]
                targets = []
                for k in range(1, self.unroll_steps + 1):
                    h_future = f['h'][info['index'] + k * self.ndt].astype(np.float32)
                    mx_future = f['mx'][info['index'] + k * self.ndt].astype(np.float32)
                    my_future = f['my'][info['index'] + k * self.ndt].astype(np.float32)
                    state = np.stack([h_future, mx_future, my_future], axis=0)
                    targets.append(state[np.newaxis, :])
                target_tensor = np.concatenate(targets, axis=0)  # [K, 3, H, W]
            else:
                # Single step: [3, H, W]
                h_next = f['h'][info['index'] + self.ndt].astype(np.float32)
                mx_next = f['mx'][info['index'] + self.ndt].astype(np.float32)
                my_next = f['my'][info['index'] + self.ndt].astype(np.float32)
                target_tensor = np.stack([h_next, mx_next, my_next], axis=0)

        # Stack as [3, H, W]
        input_tensor = np.stack([h_current, mx_current, my_current], axis=0)

        return torch.from_numpy(input_tensor), torch.from_numpy(target_tensor)


class SpinodalDecompositionDataset(Dataset):
    """
    2D Spinodal Decomposition (Phase Field) Dataset

    Fields:
        - phi_data: concentration [time, H, W] - conserved, [0, 1]

    Input: phi_t (1 channel)
    Target: phi_{t+ndt} or [phi_{t+ndt}, ..., phi_{t+K*ndt}]
    """

    def __init__(self, folder_path: str, ndt: int = 1, training_mode: str = 'onestep', unroll_steps: int = 5):
        self.folder_path = folder_path
        self.ndt = ndt
        self.training_mode = training_mode
        self.unroll_steps = unroll_steps
        self.data_info = []

        if training_mode == 'pushforward':
            required_future = ndt * unroll_steps
        else:
            required_future = ndt

        for file_path in Path(folder_path).glob("*.h5"):
            with h5py.File(file_path, 'r') as f:
                n_timesteps = f['phi_data'].shape[0]
                n_samples = n_timesteps - required_future

                for i in range(n_samples):
                    self.data_info.append({
                        'file': str(file_path),
                        'index': i
                    })

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        info = self.data_info[idx]

        with h5py.File(info['file'], 'r') as f:
            phi_current = f['phi_data'][info['index']].astype(np.float32)

            if self.training_mode == 'pushforward':
                targets = []
                for k in range(1, self.unroll_steps + 1):
                    phi_future = f['phi_data'][info['index'] + k * self.ndt].astype(np.float32)
                    targets.append(phi_future[np.newaxis, :])
                target_tensor = np.concatenate(targets, axis=0)  # [K, H, W]
            else:
                phi_next = f['phi_data'][info['index'] + self.ndt].astype(np.float32)
                target_tensor = phi_next[np.newaxis, :]  # [1, H, W]

        # Add channel dimension
        input_tensor = phi_current[np.newaxis, :]  # [1, H, W]

        return torch.from_numpy(input_tensor), torch.from_numpy(target_tensor)


def create_data_loader(
        dataset_type: str,
        folder_path: str,
        batch_size: int,
        ndt: int = 1,
        shuffle: bool = True,
        num_workers: int = 4,
        training_mode: str = 'onestep',
        unroll_steps: int = 5
) -> DataLoader:
    """
    Unified factory function for creating data loaders

    Args:
        dataset_type: One of ['convection_diffusion', 'traffic_flow', 'shallow_water', 'spinodal_decomposition']
        folder_path: Path to train/val/test folder
        batch_size: Batch size
        ndt: Number of timesteps to skip (for multi-step prediction)
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        training_mode: 'onestep' or 'pushforward'
        unroll_steps: Number of unroll steps for pushforward training

    Returns:
        DataLoader instance
    """
    # Select appropriate dataset class
    dataset_classes = {
        'convection_diffusion': ConvectionDiffusionDataset,
        'traffic_flow': TrafficFlowDataset,
        'shallow_water': ShallowWaterDataset,
        'spinodal_decomposition': SpinodalDecompositionDataset,
    }

    if dataset_type not in dataset_classes:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. "
                         f"Must be one of {list(dataset_classes.keys())}")

    # Create dataset
    dataset_class = dataset_classes[dataset_type]
    dataset = dataset_class(
        folder_path=folder_path,
        ndt=ndt,
        training_mode=training_mode,
        unroll_steps=unroll_steps
    )

    print(f"Created {dataset_type} dataset: {len(dataset)} samples, ndt={ndt}, mode={training_mode}")
    if training_mode == 'pushforward':
        print(f"  Pushforward unroll_steps={unroll_steps}")

    # Create dataloader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers if num_workers > 0 else 0,
        multiprocessing_context=get_context('spawn') if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False
    )


if __name__ == "__main__":
    """Test data loaders"""
    import os

    print("DataLoader module test")
    print("=" * 60)

    test_configs = [
        # Uncomment when you have real data
        # ('convection_diffusion', 'dataset/convection_diffusion/train'),
        # ('traffic_flow', 'dataset/traffic_flow/train'),
        # ('shallow_water', 'dataset/shallow_water/train'),
        # ('spinodal_decomposition', 'dataset/spinodal_decomposition/train'),
    ]

    for dataset_type, folder in test_configs:
        if os.path.exists(folder):
            print(f"\n=== Testing {dataset_type} loader ===")

            # Test onestep mode
            print("\n[Onestep mode]")
            loader = create_data_loader(
                dataset_type=dataset_type,
                folder_path=folder,
                batch_size=4,
                ndt=1,
                shuffle=True,
                num_workers=0,
                training_mode='onestep'
            )

            for inputs, targets in loader:
                print(f"Input shape: {inputs.shape}")
                print(f"Target shape: {targets.shape}")
                break

            # Test pushforward mode
            print("\n[Pushforward mode]")
            loader_pf = create_data_loader(
                dataset_type=dataset_type,
                folder_path=folder,
                batch_size=4,
                ndt=1,
                shuffle=True,
                num_workers=0,
                training_mode='pushforward',
                unroll_steps=5
            )

            for inputs, targets in loader_pf:
                print(f"Input shape: {inputs.shape}")
                print(f"Target shape: {targets.shape} (K={targets.shape[1] if len(targets.shape) > 2 else 'N/A'} steps)")
                break

    print("\n" + "=" * 60)
    print("DataLoader tests completed!")

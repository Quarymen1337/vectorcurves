"""
Датасеты для работы с траекториями белковых структур
"""

import torch
from torch.utils.data import Dataset
import numpy as np


class TrajectoriesDataset(Dataset):
    """
    Датасет для траекторий с нормализацией
    """
    def __init__(self, trajectories, normalize=False):
        if isinstance(trajectories[0], torch.Tensor):
            self.dataset = torch.stack(trajectories)
        else:
            self.dataset = torch.tensor(np.array(trajectories), dtype=torch.float32)

        self.normalize = normalize

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        trajectory = self.dataset[idx]

        if self.normalize:
            min_val = torch.min(trajectory)
            max_val = torch.max(trajectory)
            scale = max_val - min_val

            if scale > 1e-6:
                trajectory = (trajectory - min_val) / scale
            else:
                trajectory = torch.zeros_like(trajectory)

        # Меняем размерность с (N, 3) на (3, N) для сверточных слоев
        trajectory = trajectory.permute(1, 0)
        return trajectory, trajectory


class FixedTrajectoriesDataset(Dataset):
    """
    Исправленная версия датасета для траекторий
    """
    def __init__(self, trajectories, normalize=False):
        if isinstance(trajectories[0], torch.Tensor):
            self.dataset = torch.stack(trajectories)
        else:
            self.dataset = torch.tensor(np.array(trajectories), dtype=torch.float32)

        self.normalize = normalize

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        trajectory = self.dataset[idx]

        if self.normalize:
            min_val = torch.min(trajectory)
            max_val = torch.max(trajectory)
            scale = max_val - min_val

            if scale > 1e-6:
                trajectory = (trajectory - min_val) / scale
            else:
                trajectory = torch.zeros_like(trajectory)

            trajectory = trajectory.permute(1, 0)
            return trajectory, trajectory
        else:
            trajectory = trajectory.permute(1, 0)
            return trajectory, trajectory

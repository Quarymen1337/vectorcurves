import torch
from torch.utils.data import Dataset
import numpy as np


class FixedTrajectoriesDataset(Dataset):
    """
    Исправленная версия датасета для траекторий
    
    Основные исправления:
    1. Устранена ошибка с неопределенной переменной
    2. Правильная обработка нормализации
    3. Корректное изменение размерности для сверточных слоев
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
            trajectory_norm = trajectory.clone()
            
            min_val = torch.min(trajectory_norm)
            max_val = torch.max(trajectory_norm)
            
            scale = max_val - min_val
            
            if scale > 1e-6: 
                trajectory_norm = (trajectory_norm - min_val) / scale
            else:
                trajectory_norm = torch.zeros_like(trajectory_norm)
            
            # Изменяем порядок осей с (2048, 3) на (3, 2048) для сверточного слоя
            trajectory_norm = trajectory_norm.permute(1, 0)
            return trajectory_norm, trajectory_norm
        else:
            # Если нормализация не нужна, просто меняем размерность
            trajectory = trajectory.permute(1, 0)
            return trajectory, trajectory
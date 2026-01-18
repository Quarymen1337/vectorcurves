import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
class Trajectories_dataset(Dataset):
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

            return trajectory_norm, trajectory_norm
        
        return trajectory, trajectory, self.dataset[idx]
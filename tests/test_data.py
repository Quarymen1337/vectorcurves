"""
Тесты для модулей данных
"""

import pytest
import torch
import numpy as np
from src.data.dataset import TrajectoriesDataset, FixedTrajectoriesDataset


class TestTrajectoriesDataset:
    """Тесты для TrajectoriesDataset"""

    def test_init_with_numpy_array(self):
        """Инициализация с numpy массивом"""
        trajectories = [np.random.randn(100, 3) for _ in range(5)]
        dataset = TrajectoriesDataset(trajectories)
        assert len(dataset) == 5

    def test_init_with_tensor(self):
        """Инициализация с torch тензорами"""
        trajectories = [torch.randn(100, 3) for _ in range(5)]
        dataset = TrajectoriesDataset(trajectories)
        assert len(dataset) == 5

    def test_getitem_shape(self):
        """Проверка формы выходных данных"""
        trajectories = [np.random.randn(100, 3) for _ in range(5)]
        dataset = TrajectoriesDataset(trajectories, normalize=False)
        x, y = dataset[0]
        assert x.shape == (3, 100)
        assert y.shape == (3, 100)

    def test_normalize(self):
        """Проверка нормализации"""
        trajectories = [np.random.randn(100, 3) for _ in range(5)]
        dataset = TrajectoriesDataset(trajectories, normalize=True)
        x, y = dataset[0]
        assert x.min() >= 0
        assert x.max() <= 1


class TestFixedTrajectoriesDataset:
    """Тесты для FixedTrajectoriesDataset"""

    def test_init(self):
        """Инициализация датасета"""
        trajectories = [np.random.randn(100, 3) for _ in range(5)]
        dataset = FixedTrajectoriesDataset(trajectories)
        assert len(dataset) == 5

    def test_getitem(self):
        """Проверка получения элемента"""
        trajectories = [np.random.randn(100, 3) for _ in range(5)]
        dataset = FixedTrajectoriesDataset(trajectories, normalize=False)
        x, y = dataset[0]
        assert x.shape == (3, 100)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

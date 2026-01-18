"""
Модуль для обучения автокодировщика на кривых белковых структур.
"""

import os
import warnings
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.pdb.PDBFile import PDBFile
from src.preprocessing.interpolation import resample_trajectory_cubic_spline
from src.preprocessing.curves import get_invariant_features
from src.data.dataset import Trajectories_dataset
from src.models.rc_ae import ResConvModel
from src.metrics.diff_loss import diff_loss
from src.utils.cfg import load_config
from src.data.loader import load_and_preprocess_pdb_files





def resample_trajectories(ca_points: List[np.ndarray], 
                          k_points: int = 512) -> List[np.ndarray]:
    """
    Ресемплирует траектории CA-атомов с использованием кубических сплайнов.
    
    Args:
        ca_points: Список массивов координат CA-атомов
        k_points: Количество точек после ресемплинга
        
    Returns:
        List[np.ndarray]: Список ресемплированных траекторий
    """
    resampled_coords = []
    
    print(f"Ресемплирование траекторий до {k_points} точек...")
    
    for i, example in enumerate(tqdm(ca_points, desc="Ресемплирование")):
        try:
            # Получаем инвариантные признаки
            features = get_invariant_features(np.array(example))
            
            # Ресемплируем с помощью кубических сплайнов
            resampled = resample_trajectory_cubic_spline(features, k=k_points)
            resampled_coords.append(resampled)
            
        except Exception as e:
            print(f"Ошибка при ресемплировании примера {i}: {e}")
    
    return resampled_coords


def print_training_config(config: Dict[str, Any]) -> Tuple[str, float, int, int, int]:
    """
    Выводит параметры обучения и возвращает их значения.
    
    Args:
        config: Конфигурация проекта
        
    Returns:
        Tuple: Параметры обучения (device, learning_rate, epochs, batch_size, latent_dim)
    """
    device = config.get('model', {}).get('device', 'cpu')
    learning_rate = config.get('training', {}).get('learning_rate', 0.001)
    epochs = config.get('training', {}).get('epochs', 100)
    batch_size = config.get('training', {}).get('batch_size', 16)
    latent_dim = config.get('model', {}).get('latent_dim', 32)
    early_stop = config.get('training', {}).get('early_stoping', 10)
    
    print("\n" + "="*60)
    print("ПАРАМЕТРЫ ОБУЧЕНИЯ")
    print("="*60)
    print(f"   Learning rate: {learning_rate}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Latent dimension: {latent_dim}")
    print(f"   Early stopping patience: {early_stop}")
    print("="*60 + "\n")
    
    return device, learning_rate, epochs, batch_size, latent_dim


def train_epoch(model: ResConvModel, 
                loader: DataLoader, 
                optimizer: optim.Optimizer, 
                device: str,
                mode: str = 'train') -> Tuple[float, float]:
    """
    Выполняет одну эпоху обучения или валидации.
    
    Args:
        model: Модель автокодировщика
        loader: DataLoader с данными
        optimizer: Оптимизатор (используется только в режиме train)
        device: Устройство для вычислений
        mode: Режим работы ('train' или 'val')
        
    Returns:
        Tuple[float, float]: Средние значения total_loss и recon_loss
    """
    if mode == 'train':
        model.train()
        desc_prefix = 'Train'
    else:
        model.eval()
        desc_prefix = 'Val'
    
    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    
    progress_bar = tqdm(loader, desc=f'[{desc_prefix}]', leave=False)
    
    for batch in progress_bar:
        signals = batch[0].to(device)
        
        # Прямой проход
        if mode == 'train':
            reconstructed, latent = model(signals)
            recon_loss = diff_loss(reconstructed, signals, beta=0)
            loss = recon_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        else:
            with torch.no_grad():
                reconstructed, latent = model(signals)
                recon_loss = F.mse_loss(reconstructed, signals, reduction='mean')
                loss = recon_loss
        
        total_loss_sum += loss.item()
        recon_loss_sum += recon_loss.item()
        
        progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    avg_total_loss = total_loss_sum / len(loader)
    avg_recon_loss = recon_loss_sum / len(loader)
    
    return avg_total_loss, avg_recon_loss


def save_checkpoint(model: ResConvModel, 
                    optimizer: optim.Optimizer, 
                    epoch: int,
                    train_loss: float, 
                    val_loss: float,
                    best_val_loss: float,
                    filename: str = 'best_autoencoder.pth') -> None:
    """
    Сохраняет чекпоинт модели.
    
    Args:
        model: Модель для сохранения
        optimizer: Оптимизатор
        epoch: Номер эпохи
        train_loss: Loss на тренировочных данных
        val_loss: Loss на валидационных данных
        best_val_loss: Лучший loss на валидации
        filename: Имя файла для сохранения
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'best_val_loss': best_val_loss,
    }
    
    torch.save(checkpoint, filename)
    print(f"  ✓ Чекпоинт сохранен: {filename}")


def main():
    config = load_config()
    enable_warnings = config.get('project', {}).get('enable_warnings', False)
    if enable_warnings:
        warnings.filterwarnings('default')
        print("Предупреждения включены")
    else:
        warnings.filterwarnings('ignore')
        print("Предупреждения отключены")
    
    dataset_path = config.get('dataset', {}).get('path', 'N/A')
    pdb_files, ca_points = load_and_preprocess_pdb_files(dataset_path)
    k_points = config.get('dataset', {}).get('spline_interpolation_points', 512)
    resampled_coords = resample_trajectories(ca_points, k_points)
    val_split = config.get('training', {}).get('validation_split', 0.1)
    train_data, val_data = train_test_split(
        resampled_coords,
        test_size=val_split,
        random_state=42
    )
    
    print(f"\nРазделение данных: {len(train_data)} train, {len(val_data)} val")
    
    device, learning_rate, num_epochs, batch_size, latent_dim = print_training_config(config)
    
    train_dataset = Trajectories_dataset(train_data, normalize=True)
    val_dataset = Trajectories_dataset(val_data, normalize=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = ResConvModel(sequence_length=1024, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    train_recon_losses = []
    val_recon_losses = []
    
    patience = config.get('training', {}).get('early_stoping', 10)
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    print("\n" + "="*60)
    print("НАЧАЛО ОБУЧЕНИЯ АВТОКОДИРОВЩИКА")
    print("="*60)
    
    for epoch in range(num_epochs):
        print(f"\nЭпоха [{epoch+1}/{num_epochs}]")
        print("-" * 40)
        
        train_total_loss, train_recon_loss = train_epoch(
            model, train_loader, optimizer, device, mode='train'
        )
        
        val_total_loss, val_recon_loss = train_epoch(
            model, val_loader, optimizer, device, mode='val'
        )
        
        train_losses.append(train_total_loss)
        val_losses.append(val_total_loss)
        train_recon_losses.append(train_recon_loss)
        val_recon_losses.append(val_recon_loss)
        
        print(f"  Train - Total: {train_total_loss:.6f} | Recon: {train_recon_loss:.6f}")
        print(f"  Val   - Total: {val_total_loss:.6f} | Recon: {val_recon_loss:.6f}")
        
        if val_total_loss < best_val_loss:
            improvement = best_val_loss - val_total_loss
            best_val_loss = val_total_loss
            epochs_without_improvement = 0
            
            print(f"Новый лучший результат! Улучшение: {improvement:.6f}")
            save_checkpoint(
                model, optimizer, epoch,
                train_total_loss, val_total_loss, best_val_loss
            )
        else:
            epochs_without_improvement += 1
            print(f"  ✗ Без улучшений {epochs_without_improvement}/{patience}")
        
        if epochs_without_improvement >= patience:
            print("\n" + "!"*60)
            print(f"РАННЯЯ ОСТАНОВКА на эпохе {epoch+1}")
            print(f"Лучший val loss: {best_val_loss:.6f}")
            print("!"*60)
            break
        
        if torch.isnan(torch.tensor(val_total_loss)):
            print("Обнаружен NaN в loss, остановка обучения")
            break
    
    print("\n" + "="*60)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("="*60)
    print(f"Всего эпох: {epoch+1}")
    print(f"Лучший валидационный loss: {best_val_loss:.6f}")
    print(f"Финальный тренировочный loss: {train_losses[-1]:.6f}")
    print(f"Финальный валидационный loss: {val_losses[-1]:.6f}")
    print("="*60)
    
    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_recon_losses': train_recon_losses,
        'val_recon_losses': val_recon_losses,
        'best_val_loss': best_val_loss
    }


if __name__ == "__main__":
    results = main()
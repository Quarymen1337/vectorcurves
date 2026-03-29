"""
Модуль для обучения VAE на кривых белковых структур.
Объединяет функционал RUN_AE.py и FixedNewDataVae.ipynb
"""

import os
import warnings
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.pdb.PDBFile import PDBFile
from src.preprocessing.resample import resample_trajectories
from src.data.dataset import TrajectoriesDataset
from src.models.vae import ConvVAE
from src.models.rc_ae import ResConvModel
from src.utils.cfg import load_config
from src.data.loader import load_and_preprocess_pdb_files


def vae_loss(reconstructed, original, mu, logvar, beta=1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Функция потерь для VAE: Reconstruction Loss + KL Divergence

    Args:
        reconstructed: Реконструированные данные
        original: Оригинальные данные
        mu: Среднее латентного распределения
        logvar: Логарифм дисперсии латентного распределения
        beta: Вес KL-дивергенции

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - Общий лосс
            - Reconstruction loss
            - KL divergence
    """
    recon_loss = F.mse_loss(reconstructed, original, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kl_div
    return total_loss, recon_loss, kl_div


def train_vae_epoch(
    model: ConvVAE,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
    beta: float = 1.0,
    mode: str = 'train'
) -> Tuple[float, float, float]:
    """
    Одна эпоха обучения VAE

    Args:
        model: Модель VAE
        loader: DataLoader с данными
        optimizer: Оптимизатор
        device: Устройство для вычислений
        beta: Вес KL-дивергенции
        mode: Режим работы ('train' или 'val')

    Returns:
        Tuple[float, float, float]: Средние значения total_loss, recon_loss, kl_div
    """
    if mode == 'train':
        model.train()
        desc_prefix = 'Train'
    else:
        model.eval()
        desc_prefix = 'Val'

    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    kl_loss_sum = 0.0

    progress_bar = tqdm(loader, desc=f'[{desc_prefix}]', leave=False)

    for batch in progress_bar:
        if isinstance(batch, (list, tuple)):
            signals = batch[0].to(device)
        else:
            signals = batch.to(device)

        if mode == 'train':
            reconstructed, mu, logvar, z = model(signals)
            loss, recon_loss, kl_loss = vae_loss(reconstructed, signals, mu, logvar, beta)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        else:
            with torch.no_grad():
                reconstructed, mu, logvar, z = model(signals)
                loss, recon_loss, kl_loss = vae_loss(reconstructed, signals, mu, logvar, beta)

        total_loss_sum += loss.item()
        recon_loss_sum += recon_loss.item()
        kl_loss_sum += kl_loss.item()

        progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

    avg_total_loss = total_loss_sum / len(loader)
    avg_recon_loss = recon_loss_sum / len(loader)
    avg_kl_loss = kl_loss_sum / len(loader)

    return avg_total_loss, avg_recon_loss, avg_kl_loss


def train_ae_epoch(
    model: ResConvModel,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
    mode: str = 'train'
) -> Tuple[float, float]:
    """
    Одна эпоха обучения AutoEncoder

    Args:
        model: Модель AE
        loader: DataLoader с данными
        optimizer: Оптимизатор
        device: Устройство для вычислений
        mode: Режим работы ('train' или 'val')

    Returns:
        Tuple[float, float]: Средние значения total_loss, recon_loss
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
        if isinstance(batch, (list, tuple)):
            signals = batch[0].to(device)
        else:
            signals = batch.to(device)

        if mode == 'train':
            reconstructed, latent = model(signals)
            recon_loss = F.mse_loss(reconstructed, signals, reduction='sum')
            loss = recon_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        else:
            with torch.no_grad():
                reconstructed, latent = model(signals)
                recon_loss = F.mse_loss(reconstructed, signals, reduction='sum')
                loss = recon_loss

        total_loss_sum += loss.item()
        recon_loss_sum += recon_loss.item()

        progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

    avg_total_loss = total_loss_sum / len(loader)
    avg_recon_loss = recon_loss_sum / len(loader)

    return avg_total_loss, avg_recon_loss


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    best_val_loss: float,
    filename: str = 'best_model.pth',
    is_vae: bool = False
) -> None:
    """
    Сохраняет чекпоинт модели

    Args:
        model: Модель для сохранения
        optimizer: Оптимизатор
        epoch: Номер эпохи
        train_loss: Loss на тренировочных данных
        val_loss: Loss на валидационных данных
        best_val_loss: Лучший loss на валидации
        filename: Имя файла для сохранения
        is_vae: Флаг VAE модели
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'best_val_loss': best_val_loss,
        'model_type': 'VAE' if is_vae else 'AE'
    }

    torch.save(checkpoint, filename)
    print(f"✓ Чекпоинт сохранен: {filename}")


def train_vae(
    dataset_path: str,
    config: Optional[Dict[str, Any]] = None,
    use_vae: bool = True
) -> Dict[str, Any]:
    """
    Основная функция обучения VAE/AE

    Args:
        dataset_path: Путь к датасету с PDB файлами
        config: Конфигурация обучения
        use_vae: Использовать VAE или AE

    Returns:
        Dict[str, Any]: Результаты обучения
    """
    # Загрузка конфигурации
    if config is None:
        config = load_config()

    # Параметры
    enable_warnings = config.get('project', {}).get('enable_warnings', False)
    if enable_warnings:
        warnings.filterwarnings('default')
    else:
        warnings.filterwarnings('ignore')

    # Загрузка данных
    print("\n" + "="*60)
    print("ЗАГРУЗКА ДАННЫХ")
    print("="*60)

    pdb_files, ca_points = load_and_preprocess_pdb_files(dataset_path)

    k_points = config.get('dataset', {}).get('spline_interpolation_points', 512)
    print(f"Ресемплинг к {k_points} точкам...")
    resampled_coords = resample_trajectories(ca_points, k_points)

    val_split = config.get('training', {}).get('validation_split', 0.1)
    train_data, val_data = train_test_split(
        resampled_coords,
        test_size=val_split,
        random_state=42
    )

    print(f"Разделение данных: {len(train_data)} train, {len(val_data)} val")

    # Параметры обучения
    device = config.get('model', {}).get('device', 'cpu')
    learning_rate = config.get('training', {}).get('learning_rate', 0.001)
    num_epochs = config.get('training', {}).get('epochs', 100)
    batch_size = config.get('training', {}).get('batch_size', 32)
    latent_dim = config.get('model', {}).get('latent_dim', 64)
    patience = config.get('training', {}).get('early_stoping', 15)

    print(f"\nУстройство: {device}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Latent dimension: {latent_dim}")
    print(f"Early stopping patience: {patience}")

    # Создание датасетов
    train_dataset = TrajectoriesDataset(train_data, normalize=True)
    val_dataset = TrajectoriesDataset(val_data, normalize=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Создание модели
    sequence_length = k_points
    if use_vae:
        model = ConvVAE(
            input_channels=3,
            sequence_length=sequence_length,
            latent_dim=latent_dim
        ).to(device)
        print(f"\nСоздана VAE модель")
    else:
        model = ResConvModel(
            input_channels=3,
            sequence_length=sequence_length,
            latent_dim=latent_dim
        ).to(device)
        print(f"\nСоздана AE модель")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # История обучения
    train_losses = []
    val_losses = []
    train_recon_losses = []
    val_recon_losses = []
    train_kl_losses = [] if use_vae else None
    val_kl_losses = [] if use_vae else None
    beta_values = []

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    beta = 0.0  # Начальное значение beta для VAE

    print("\n" + "="*60)
    print(f"НАЧАЛО ОБУЧЕНИЯ {'VAE' if use_vae else 'AE'}")
    print("="*60)

    for epoch in range(num_epochs):
        # Annealing beta для VAE
        if use_vae:
            beta = min(1.0, epoch / 50)  # Постепенное увеличение beta
            beta_values.append(beta)

        print(f"\nЭпоха [{epoch+1}/{num_epochs}]")
        print("-" * 40)

        if use_vae:
            train_total, train_recon, train_kl = train_vae_epoch(
                model, train_loader, optimizer, device, beta, mode='train'
            )
            val_total, val_recon, val_kl = train_vae_epoch(
                model, val_loader, optimizer, device, beta, mode='val'
            )

            train_losses.append(train_total)
            val_losses.append(val_total)
            train_recon_losses.append(train_recon)
            val_recon_losses.append(val_recon)
            train_kl_losses.append(train_kl)
            val_kl_losses.append(val_kl)

            print(f"  Train - Total: {train_total:.6f} | Recon: {train_recon:.6f} | KL: {train_kl:.6f}")
            print(f"  Val   - Total: {val_total:.6f} | Recon: {val_recon:.6f} | KL: {val_kl:.6f}")
        else:
            train_total, train_recon = train_ae_epoch(
                model, train_loader, optimizer, device, mode='train'
            )
            val_total, val_recon = train_ae_epoch(
                model, val_loader, optimizer, device, mode='val'
            )

            train_losses.append(train_total)
            val_losses.append(val_total)
            train_recon_losses.append(train_recon)
            val_recon_losses.append(val_recon)

            print(f"  Train - Total: {train_total:.6f} | Recon: {train_recon:.6f}")
            print(f"  Val   - Total: {val_total:.6f} | Recon: {val_recon:.6f}")

        # Проверка на улучшение
        if val_total < best_val_loss:
            improvement = best_val_loss - val_total
            best_val_loss = val_total
            epochs_without_improvement = 0

            print(f"✨ Новый лучший результат! Улучшение: {improvement:.6f}")
            save_checkpoint(
                model, optimizer, epoch,
                train_total, val_total, best_val_loss,
                filename='best_vae_model.pth' if use_vae else 'best_ae_model.pth',
                is_vae=use_vae
            )
        else:
            epochs_without_improvement += 1
            print(f"  ✗ Без улучшений {epochs_without_improvement}/{patience}")

        # Ранняя остановка
        if epochs_without_improvement >= patience:
            print("\n" + "!"*60)
            print(f"РАННЯЯ ОСТАНОВКА на эпохе {epoch+1}")
            print(f"Лучший val loss: {best_val_loss:.6f}")
            print("!"*60)
            break

        # Проверка на NaN
        if torch.isnan(torch.tensor(val_total)):
            print("⚠️  Обнаружен NaN в loss, остановка обучения")
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
        'train_kl_losses': train_kl_losses,
        'val_kl_losses': val_kl_losses,
        'beta_values': beta_values,
        'best_val_loss': best_val_loss
    }


def main():
    """
    Точка входа для обучения
    """
    config = load_config()
    dataset_path = config.get('dataset', {}).get('path', 'data/dompdb')

    # Обучение VAE
    print("\n" + "="*80)
    print(" " * 25 + "ОБУЧЕНИЕ VAE")
    print("="*80)

    vae_results = train_vae(dataset_path, config, use_vae=True)

    # При желании можно обучить AE
    # print("\n" + "="*80)
    # print(" " * 25 + "ОБУЧЕНИЕ AE")
    # print("="*80)
    # ae_results = train_vae(dataset_path, config, use_vae=False)

    return vae_results


if __name__ == "__main__":
    results = main()

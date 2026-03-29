"""
Скрипт для анализа латентного пространства VAE
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA


def analyze_latent_space_statistics(model, dataloader, device, n_samples=1000):
    """
    Функция для анализа статистик латентного пространства
    
    Параметры:
    - model: обученная модель VAE
    - dataloader: загрузчик данных
    - device: устройство для вычислений
    - n_samples: количество образцов для анализа
    """
    model.eval()
    
    all_latents = []
    
    with torch.no_grad():
        count = 0
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x, _ = batch
            else:
                x = batch
            
            x = x.to(device)
            mu, logvar = model.encode(x)
            
            batch_size = min(len(mu), n_samples - count)
            all_latents.append(mu[:batch_size].cpu().numpy())
            
            count += batch_size
            if count >= n_samples:
                break
    
    all_latents = np.vstack(all_latents)
    
    # Вычисление статистик
    mean_per_dim = np.mean(all_latents, axis=0)
    var_per_dim = np.var(all_latents, axis=0)
    std_per_dim = np.std(all_latents, axis=0)
    
    print(f"Размерность латентного пространства: {all_latents.shape[1]}")
    print(f"Количество образцов: {all_latents.shape[0]}")
    print(f"Среднее значение по всем размерностям: {np.mean(mean_per_dim):.4f}")
    print(f"Средняя дисперсия по всем размерностям: {np.mean(var_per_dim):.4f}")
    print(f"Среднее стандартное отклонение по всем размерностям: {np.mean(std_per_dim):.4f}")
    
    # Проверка на posterior collapse
    dims_with_low_var = np.where(var_per_dim < 0.1)[0]
    print(f"Количество размерностей с низкой дисперсией (< 0.1): {len(dims_with_low_var)}")
    
    if len(dims_with_low_var) > all_latents.shape[1] // 2:
        print("ПРЕДУПРЕЖДЕНИЕ: Большое количество размерностей имеют низкую дисперсию - возможен posterior collapse!")
    else:
        print("Латентное пространство используется достаточно эффективно")
    
    # Визуализация дисперсии по размерностям
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(var_per_dim)
    plt.title('Дисперсия по размерностям латентного пространства')
    plt.xlabel('Размерность')
    plt.ylabel('Дисперсия')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.hist(var_per_dim, bins=50, alpha=0.7, edgecolor='black')
    plt.title('Распределение дисперсии по размерностям')
    plt.xlabel('Дисперсия')
    plt.ylabel('Количество размерностей')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(std_per_dim)
    plt.title('Стандартное отклонение по размерностям')
    plt.xlabel('Размерность')
    plt.ylabel('Стандартное отклонение')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # PCA анализ
    pca = PCA()
    pca.fit(all_latents)
    
    # Объясненная дисперсия
    cumsum_var_ratio = np.cumsum(pca.explained_variance_ratio_)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(pca.explained_variance_ratio_[:50])  # Показываем первые 50 компонент
    plt.title('Объясненная дисперсия по главным компонентам (первые 50)')
    plt.xlabel('Главные компоненты')
    plt.ylabel('Объясненная дисперсия')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(cumsum_var_ratio[:100])  # Показываем первые 100 компонент
    plt.title('Кумулятивная объясненная дисперсия (первые 100)')
    plt.xlabel('Количество компонент')
    plt.ylabel('Кумулятивная объясненная дисперсия')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Объясненная дисперсия первыми 10 компонентами: {cumsum_var_ratio[9]:.4f}")
    print(f"Объясненная дисперсия первыми 50 компонентами: {cumsum_var_ratio[49]:.4f}")
    print(f"Объясненная дисперсия первыми 100 компонентами: {cumsum_var_ratio[99]:.4f}")
    
    return {
        'latents': all_latents,
        'mean_per_dim': mean_per_dim,
        'var_per_dim': var_per_dim,
        'std_per_dim': std_per_dim,
        'dims_with_low_var': dims_with_low_var,
        'pca': pca,
        'cumsum_var_ratio': cumsum_var_ratio
    }


def check_posterior_collapse(model, dataloader, device, n_samples=1000, threshold=0.1):
    """
    Функция для проверки posterior collapse
    
    Параметры:
    - model: обученная модель VAE
    - dataloader: загрузчик данных
    - device: устройство для вычислений
    - n_samples: количество образцов для анализа
    - threshold: порог для определения low-variance dimensions
    """
    model.eval()
    
    all_latents = []
    all_logvars = []
    
    with torch.no_grad():
        count = 0
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x, _ = batch
            else:
                x = batch
            
            x = x.to(device)
            mu, logvar = model.encode(x)
            
            batch_size = min(len(mu), n_samples - count)
            all_latents.append(mu[:batch_size].cpu().numpy())
            all_logvars.append(logvar[:batch_size].cpu().numpy())
            
            count += batch_size
            if count >= n_samples:
                break
    
    all_latents = np.vstack(all_latents)
    all_logvars = np.vstack(all_logvars)
    
    # Вычисление дисперсии в латентном пространстве
    var_per_dim = np.var(all_latents, axis=0)
    mean_logvar_per_dim = np.mean(all_logvars, axis=0)
    
    # Вычисление KL-дивергенции для каждой размерности
    kl_per_dim = -0.5 * (1 + mean_logvar_per_dim - np.mean(all_latents**2, axis=0) - np.exp(mean_logvar_per_dim))
    
    print("=== АНАЛИЗ POSTERIOR COLLAPSE ===")
    print(f"Средняя дисперсия по всем размерностям: {np.mean(var_per_dim):.4f}")
    print(f"Среднее exp(logvar) по всем размерностям: {np.mean(np.exp(mean_logvar_per_dim)):.4f}")
    print(f"Средняя KL-дивергенция по размерностям: {np.mean(kl_per_dim):.4f}")
    
    # Подсчет размерностей с признаками collapse
    low_var_dims = np.sum(var_per_dim < threshold)
    low_kl_dims = np.sum(kl_per_dim < 0.01)  # Размерности с очень низкой KL
    
    print(f"Размерностей с низкой дисперсией (< {threshold}): {low_var_dims}/{all_latents.shape[1]} ({low_var_dims/all_latents.shape[1]*100:.1f}%)")
    print(f"Размерностей с низкой KL-дивергенцией (< 0.01): {low_kl_dims}/{all_latents.shape[1]} ({low_kl_dims/all_latents.shape[1]*100:.1f}%)")
    
    if low_var_dims > all_latents.shape[1] * 0.5:
        print("⚠️  СЕРЬЕЗНОЕ ПРЕДУПРЕЖДЕНИЕ: Более 50% размерностей имеют низкую дисперсию - сильный признак posterior collapse!")
    elif low_var_dims > all_latents.shape[1] * 0.2:
        print("⚠️  ПРЕДУПРЕЖДЕНИЕ: Более 20% размерностей имеют низкую дисперсию - возможен частичный posterior collapse.")
    else:
        print("✅  Posterior collapse маловероятен - большинство размерностей активно используются.")
    
    return {
        'var_per_dim': var_per_dim,
        'mean_logvar_per_dim': mean_logvar_per_dim,
        'kl_per_dim': kl_per_dim,
        'low_var_dims': low_var_dims,
        'low_kl_dims': low_kl_dims
    }


if __name__ == "__main__":
    print("Этот скрипт содержит функции для анализа латентного пространства VAE.")
    print("Для использования загрузите обученную модель и данные.")
"""
Скрипт для t-SNE визуализации латентного пространства VAE
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
from mpl_toolkits.mplot3d import Axes3D


def visualize_latent_space_tsne_1(model, dataloader, device, n_samples=1000, perplexity=30, n_components=2, random_state=42):
    """
    Функция для t-SNE визуализации латентного пространства
    
    Параметры:
    - model: обученная модель VAE
    - dataloader: загрузчик данных
    - device: устройство для вычислений
    - n_samples: количество образцов для анализа
    - perplexity: параметр t-SNE, обычно от 5 до 50
    - n_components: количество компонент для возврата (обычно 2 для визуализации)
    - random_state: для воспроизводимости
    """
    model.eval()
    
    all_latents = []
    all_samples = []
    
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
            all_samples.append(x[:batch_size].cpu().numpy())
            
            count += batch_size
            if count >= n_samples:
                break
    
    all_latents = np.vstack(all_latents)
    all_samples = np.vstack(all_samples)
    
    print(f"Выполняется t-SNE для {all_latents.shape[0]} образцов из {all_latents.shape[1]}-мерного пространства...")
    
    # Применение t-SNE
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    
    latents_tsne = tsne.fit_transform(all_latents)
    
    # Визуализация
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(latents_tsne[:, 0], latents_tsne[:, 1], alpha=0.6, s=20)
    plt.title(f't-SNE визуализация латентного пространства\n(perplexity={perplexity})')
    plt.xlabel('t-SNE компонента 1')
    plt.ylabel('t-SNE компонента 2')
    plt.grid(True, alpha=0.3)
    plt.colorbar(scatter)
    plt.show()
    
    # Также покажем PCA для сравнения
    pca = PCA(n_components=2)
    latents_pca = pca.fit_transform(all_latents)
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(latents_pca[:, 0], latents_pca[:, 1], alpha=0.6, s=20)
    plt.title(f'PCA визуализация латентного пространства\n(Объясненная дисперсия: {pca.explained_variance_ratio_.sum():.3f})')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True, alpha=0.3)
    plt.colorbar(scatter)
    plt.show()
    
    print(f"Доля объясненной дисперсии первыми двумя компонентами (PCA): {pca.explained_variance_ratio_.sum():.4f}")
    
    return latents_tsne, latents_pca, all_latents


def analyze_latent_space_detailed(model, dataloader, device, n_samples=500):
    """
    Подробный анализ латентного пространства
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
    
    # Статистики латентного пространства
    print(f"Размерность латентного пространства: {all_latents.shape[1]}")
    print(f"Количество образцов: {all_latents.shape[0]}")
    print(f"Среднее значение по каждой размерности: {np.mean(all_latents, axis=0)[:5]}...")  # Показываем первые 5
    print(f"Стандартное отклонение по каждой размерности: {np.std(all_latents, axis=0)[:5]}...")  # Показываем первые 5
    print(f"Диапазон значений: [{np.min(all_latents):.4f}, {np.max(all_latents):.4f}]")
    
    # Гистограммы нескольких размерностей
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i in range(min(6, all_latents.shape[1])):
        row = i // 3
        col = i % 3
        axes[row, col].hist(all_latents[:, i], bins=50, alpha=0.7, edgecolor='black')
        axes[row, col].set_title(f'Размерность {i}')
        axes[row, col].grid(True, alpha=0.3)
    
    # Удаляем лишние subplot'ы если размерностей меньше 6
    for i in range(all_latents.shape[1], 6):
        if i >= all_latents.shape[1]:
            row = i // 3
            col = i % 3
            if row < 2 and col < 3:
                fig.delaxes(axes[row, col])
    
    plt.tight_layout()
    plt.show()
    
    return all_latents


def plot_multiple_perplexities(model, dataloader, device, n_samples=300, perplexities=[5, 15, 30, 50]):
    """
    Визуализация с несколькими значениями perplexity для сравнения
    """
    model.eval()
    
    # Получаем латентные представления
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
    
    # Создаем сетку для визуализации
    n_perplexities = len(perplexities)
    cols = 2
    rows = int(np.ceil(n_perplexities / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 8 * rows))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for idx, p in enumerate(perplexities):
        if idx >= len(axes):
            break
            
        tsne = TSNE(
            n_components=2,
            perplexity=p,
            random_state=42,
            n_jobs=-1
        )
        
        latents_tsne = tsne.fit_transform(all_latents)
        
        axes[idx].scatter(latents_tsne[:, 0], latents_tsne[:, 1], alpha=0.6, s=20)
        axes[idx].set_title(f't-SNE с perplexity={p}')
        axes[idx].grid(True, alpha=0.3)
    
    # Удаляем лишние subplot'ы
    for idx in range(len(perplexities), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Этот скрипт содержит функции для t-SNE визуализации латентного пространства VAE.")
    print("Для использования загрузите обученную модель и данные.")
"""
Скрипт для анализа результатов обучения VAE
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from curve_comparison import compare_curves_3d, calculate_reconstruction_metrics


def analyze_training_history(history):
    """
    Функция для анализа истории обучения
    
    Параметры:
    - history: словарь с историей обучения из функции improved_train_vae
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Общий лосс
    axes[0, 0].plot(history['train_loss'], label='Train Loss', alpha=0.7)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', alpha=0.7)
    axes[0, 0].set_title('Общий лосс')
    axes[0, 0].set_xlabel('Эпоха')
    axes[0, 0].set_ylabel('Значение лосса')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Реконструкция
    axes[0, 1].plot(history['train_recon'], label='Train Recon', alpha=0.7)
    axes[0, 1].plot(history['val_recon'], label='Val Recon', alpha=0.7)
    axes[0, 1].set_title('Реконструкционная ошибка')
    axes[0, 1].set_xlabel('Эпоха')
    axes[0, 1].set_ylabel('Значение ошибки')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # KL-дивергенция
    axes[0, 2].plot(history['train_kl'], label='Train KL', alpha=0.7)
    axes[0, 2].plot(history['val_kl'], label='Val KL', alpha=0.7)
    axes[0, 2].set_title('KL-дивергенция')
    axes[0, 2].set_xlabel('Эпоха')
    axes[0, 2].set_ylabel('Значение KL')
    axes[0, 2].legend()
    axes[0, 2].grid(True)

    # Beta значения
    axes[1, 0].plot(history['beta_values'], label='Beta values', color='red', alpha=0.7)
    axes[1, 0].set_title('Значения Beta')
    axes[1, 0].set_xlabel('Эпоха')
    axes[1, 0].set_ylabel('Значение Beta')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Сравнение реконструкции и KL
    axes[1, 1].plot(history['train_recon'], label='Train Recon', alpha=0.7)
    axes[1, 1].plot(history['train_kl'], label='Train KL', alpha=0.7)
    axes[1, 1].set_title('Сравнение Recon и KL (Train)')
    axes[1, 1].set_xlabel('Эпоха')
    axes[1, 1].set_ylabel('Значение')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # Сравнение реконструкции и KL (Validation)
    axes[1, 2].plot(history['val_recon'], label='Val Recon', alpha=0.7)
    axes[1, 2].plot(history['val_kl'], label='Val KL', alpha=0.7)
    axes[1, 2].set_title('Сравнение Recon и KL (Validation)')
    axes[1, 2].set_xlabel('Эпоха')
    axes[1, 2].set_ylabel('Значение')
    axes[1, 2].legend()
    axes[1, 2].grid(True)

    plt.tight_layout()
    plt.show()


def evaluate_model_quality(model, dataloader, device, num_examples=5):
    """
    Функция для оценки качества модели на нескольких примерах
    
    Параметры:
    - model: обученная модель VAE
    - dataloader: загрузчик данных
    - device: устройство для вычислений
    - num_examples: количество примеров для анализа
    """
    model.eval()
    all_metrics = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, (list, tuple)):
                sample_data, _ = batch
            else:
                sample_data = batch
            
            sample_data = sample_data.to(device)
            reconstructed, mu, logvar = model(sample_data)
            
            # Вычисление метрик для каждого примера в батче
            for i in range(min(len(sample_data), num_examples - len(all_metrics))):
                if len(all_metrics) >= num_examples:
                    break
                    
                original = sample_data[i].permute(1, 0).cpu().numpy()
                reconstructed_curve = reconstructed[i].permute(1, 0).cpu().numpy()
                
                metrics = calculate_reconstruction_metrics(original, reconstructed_curve)
                all_metrics.append(metrics)
                
                # Визуализация
                fig, ax = compare_curves_3d(
                    original,
                    reconstructed_curve,
                    title=f'Пример {len(all_metrics)}: Оригинал vs Реконструкция',
                    labels=('Оригинал', 'Реконструкция')
                )
                plt.show()
                
                print(f"Метрики для примера {len(all_metrics)}:")
                for metric, value in metrics.items():
                    print(f"  {metric}: {value:.6f}")
                print("-" * 40)
            
            if len(all_metrics) >= num_examples:
                break
    
    # Средние метрики
    avg_metrics = {}
    for metric in all_metrics[0].keys():
        avg_metrics[metric] = np.mean([m[metric] for m in all_metrics])
    
    print("\nСредние метрики по всем примерам:")
    for metric, value in avg_metrics.items():
        print(f"  {metric}: {value:.6f}")
    
    return all_metrics, avg_metrics


def analyze_latent_space(model, dataloader, device, n_samples=1000):
    """
    Функция для анализа латентного пространства

    Параметры:
    - model: обученная модель VAE
    - dataloader: загрузчик данных
    - device: устройство для вычислений
    - n_samples: количество образцов для анализа
    """
    model.eval()

    all_latents = []
    all_labels = []

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
            all_labels.extend(list(range(count, count + batch_size)))

            count += batch_size
            if count >= n_samples:
                break

    all_latents = np.vstack(all_latents)

    # Визуализация распределения в латентном пространстве
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Первые две размерности
    axes[0].scatter(all_latents[:, 0], all_latents[:, 1], alpha=0.6)
    axes[0].set_title('Распределение в первых двух размерностях латентного пространства')
    axes[0].set_xlabel('Latent dim 0')
    axes[0].set_ylabel('Latent dim 1')
    axes[0].grid(True)

    # Гистограмма значений в одной размерности
    axes[1].hist(all_latents[:, 0], bins=50, alpha=0.7, edgecolor='black')
    axes[1].set_title('Гистограмма значений в первой размерности латентного пространства')
    axes[1].set_xlabel('Значение')
    axes[1].set_ylabel('Частота')
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

    # Статистики латентного пространства
    print(f"Размерность латентного пространства: {all_latents.shape[1]}")
    print(f"Среднее значение по каждой размерности: {np.mean(all_latents, axis=0)[:5]}...")  # Показываем первые 5
    print(f"Стандартное отклонение по каждой размерности: {np.std(all_latents, axis=0)[:5]}...")  # Показываем первые 5
    print(f"Диапазон значений: [{np.min(all_latents):.4f}, {np.max(all_latents):.4f}]")

    return all_latents


def tsne_visualization(model, dataloader, device, n_samples=1000, perplexity=30, n_components=2, random_state=42):
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
    from sklearn.manifold import TSNE

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

    print(f"Выполняется t-SNE для {all_latents.shape[0]} образцов из {all_latents.shape[1]}-мерного пространства...")

    # Применение t-SNE
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        n_jobs=-1,  # Использовать все доступные ядра
        verbose=1
    )

    latents_tsne = tsne.fit_transform(all_latents)

    # Визуализация
    plt.figure(figsize=(10, 8))
    plt.scatter(latents_tsne[:, 0], latents_tsne[:, 1], alpha=0.6)
    plt.title(f't-SNE визуализация латентного пространства\n(perplexity={perplexity})')
    plt.xlabel('t-SNE компонента 1')
    plt.ylabel('t-SNE компонента 2')
    plt.grid(True)
    plt.show()

    # Также можем показать несколько примеров с разными perplexity
    if n_samples >= 300:  # Если достаточно данных
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        perplexities = [5, 15, 30, 50]

        for idx, p in enumerate(perplexities):
            ax = axes[idx // 2, idx % 2]

            tsne_temp = TSNE(
                n_components=2,
                perplexity=p,
                random_state=random_state,
                n_jobs=-1
            )

            latents_tsne_temp = tsne_temp.fit_transform(all_latents[:300])  # Используем меньше данных для скорости

            ax.scatter(latents_tsne_temp[:, 0], latents_tsne_temp[:, 1], alpha=0.6)
            ax.set_title(f't-SNE с perplexity={p}')
            ax.grid(True)

        plt.tight_layout()
        plt.show()

    return latents_tsne


def umap_visualization(model, dataloader, device, n_samples=1000, n_neighbors=15, min_dist=0.1, random_state=42):
    """
    Функция для UMAP визуализации латентного пространства

    Параметры:
    - model: обученная модель VAE
    - dataloader: загрузчик данных
    - device: устройство для вычислений
    - n_samples: количество образцов для анализа
    - n_neighbors: количество соседей в UMAP
    - min_dist: минимальное расстояние между точками
    - random_state: для воспроизводимости
    """
    try:
        from umap import UMAP
    except ImportError:
        print("UMAP не установлен. Установите с помощью: pip install umap-learn")
        return None

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

    print(f"Выполняется UMAP для {all_latents.shape[0]} образцов из {all_latents.shape[1]}-мерного пространства...")

    # Применение UMAP
    reducer = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state
    )

    latents_umap = reducer.fit_transform(all_latents)

    # Визуализация
    plt.figure(figsize=(10, 8))
    plt.scatter(latents_umap[:, 0], latents_umap[:, 1], alpha=0.6)
    plt.title(f'UMAP визуализация латентного пространства\n(n_neighbors={n_neighbors}, min_dist={min_dist})')
    plt.xlabel('UMAP компонента 1')
    plt.ylabel('UMAP компонента 2')
    plt.grid(True)
    plt.show()

    return latents_umap


def generate_and_visualize(model, device, num_samples=5):
    """
    Функция для генерации новых образцов и их визуализации
    
    Параметры:
    - model: обученная модель VAE
    - device: устройство для вычислений
    - num_samples: количество образцов для генерации
    """
    model.eval()
    
    # Генерация случайных векторов из нормального распределения
    random_latents = torch.randn(num_samples, model.latent_dim).to(device)
    
    # Декодирование в пространство признаков
    generated_samples = model.decode(random_latents)
    
    # Визуализация сгенерированных образцов
    for i in range(num_samples):
        generated_curve = generated_samples[i].permute(1, 0).cpu().numpy()
        
        fig, ax = compare_curves_3d(
            generated_curve,
            title=f'Сгенерированная кривая {i+1}',
            labels=('Сгенерированная кривая', '')
        )
        plt.show()


# Пример использования функций (когда доступна обученная модель)
if __name__ == "__main__":
    print("Файл содержит функции для анализа результатов обучения VAE.")
    print("Для использования этих функций загрузите обученную модель и историю обучения.")
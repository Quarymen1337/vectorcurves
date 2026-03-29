"""
Скрипт для анализа обученной модели и визуализации результатов
"""

import argparse
import torch
import matplotlib.pyplot as plt
from src.utils.cfg import load_config
from src.models.vae import ConvVAE
from src.models.rc_ae import ResConvModel
from src.analysis import (
    analyze_training_history,
    analyze_latent_space_statistics,
    check_posterior_collapse,
    visualize_latent_space_tsne_1
)


def load_model(model_path: str, config: dict, device: str):
    """Загрузка обученной модели"""
    checkpoint = torch.load(model_path, map_location=device)
    
    latent_dim = config.get('model', {}).get('latent_dim', 64)
    sequence_length = config.get('dataset', {}).get('spline_interpolation_points', 512)
    
    model_type = checkpoint.get('model_type', 'VAE')
    
    if model_type == 'VAE':
        model = ConvVAE(
            input_channels=3,
            sequence_length=sequence_length,
            latent_dim=latent_dim
        ).to(device)
    else:
        model = ResConvModel(
            input_channels=3,
            sequence_length=sequence_length,
            latent_dim=latent_dim
        ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Загружена модель: {model_path}")
    print(f"  Тип: {model_type}")
    print(f"  Эпоха: {checkpoint['epoch']}")
    print(f"  Best val loss: {checkpoint['best_val_loss']:.6f}")
    
    return model, model_type


def plot_training_history(results: dict, save_path: str = None):
    """Визуализация истории обучения"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Общий loss
    axes[0, 0].plot(results['train_losses'], label='Train')
    axes[0, 0].plot(results['val_losses'], label='Val')
    axes[0, 0].set_title('Общий loss')
    axes[0, 0].set_xlabel('Эпоха')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Reconstruction loss
    axes[0, 1].plot(results['train_recon_losses'], label='Train Recon')
    axes[0, 1].plot(results['val_recon_losses'], label='Val Recon')
    axes[0, 1].set_title('Reconstruction loss')
    axes[0, 1].set_xlabel('Эпоха')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # KL loss (если есть)
    if 'train_kl_losses' in results and results['train_kl_losses'] is not None:
        axes[1, 0].plot(results['train_kl_losses'], label='Train KL')
        axes[1, 0].plot(results['val_kl_losses'], label='Val KL')
        axes[1, 0].set_title('KL divergence')
        axes[1, 0].set_xlabel('Эпоха')
        axes[1, 0].set_ylabel('KL')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Beta values (если есть)
    if 'beta_values' in results and results['beta_values']:
        axes[1, 1].plot(results['beta_values'], color='red')
        axes[1, 1].set_title('Beta value (KL weight)')
        axes[1, 1].set_xlabel('Эпоха')
        axes[1, 1].set_ylabel('Beta')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ График сохранен: {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Анализ обученной модели VAE/AE')
    parser.add_argument('--model', type=str, default='best_vae_model.pth',
                        help='Путь к файлу модели')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Путь к конфигурационному файлу')
    parser.add_argument('--analyze-latent', action='store_true',
                        help='Анализировать латентное пространство')
    parser.add_argument('--plot-history', action='store_true',
                        help='Построить график истории обучения')
    parser.add_argument('--output', type=str, default=None,
                        help='Путь для сохранения результатов')
    
    args = parser.parse_args()
    
    # Загрузка конфигурации
    config = load_config()
    device = config.get('model', {}).get('device', 'cpu')
    
    # Загрузка модели
    model, model_type = load_model(args.model, config, device)
    
    print("\n" + "="*60)
    print("АНАЛИЗ МОДЕЛИ")
    print("="*60)
    
    # Анализ латентного пространства для VAE
    if args.analyze_latent and model_type == 'VAE':
        print("\nАнализ латентного пространства...")
        # Требуется dataloader для анализа
        # analyze_latent_space_statistics(model, dataloader, device)
        # check_posterior_collapse(model, dataloader, device)
        print("Для полного анализа требуется загрузить данные")
    
    print("\n✓ Анализ завершен")


if __name__ == "__main__":
    main()

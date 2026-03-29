import torch
import torch.nn as nn
from improved_vae import ImprovedConv1DVAEWithClassConditioning
from multitask_loss import MultitaskVAEWithClassConditioningLoss, compute_per_point_accuracy


def example_usage():
    """
    Пример использования улучшенной модели VAE с условием класса для каждой точки
    """
    # Гиперпараметры
    latent_dim = 128
    input_channels = 3  # 3 канала для кривых
    sequence_length = 1024  # как вы указали: [4, 1024] - 1024 точки на кривой
    batch_size = 4  # как в вашем примере: [4, 1024] - 4 кривые в батче
    
    # Создаем модель
    model = ImprovedConv1DVAEWithClassConditioning(
        latent_dim=latent_dim,
        input_channels=input_channels,
        sequence_length=sequence_length
    )
    
    # Пример данных
    # x.shape = (batch_size, input_channels, sequence_length)
    x = torch.randn(batch_size, input_channels, sequence_length)
    
    # Маска классов - для каждой точки кривой (0 или 1)
    # class_mask.shape = (batch_size, sequence_length)
    class_mask = torch.randint(0, 2, (batch_size, sequence_length)).float()
    
    # Целевые метки классов для обучения (те же самые, что и маска)
    class_targets = class_mask.long()
    
    # Прямой проход
    reconstructed_curves, mu, logvar, class_logits, class_probs = model(x, class_mask)
    
    # Вычисляем потери
    criterion = MultitaskVAEWithClassConditioningLoss(kl_weight=1.0, recon_weight=1.0, class_weight=1.0)
    total_loss, recon_loss, kl_loss, class_loss = criterion(
        reconstructed_curves, x, mu, logvar, class_logits, class_targets
    )
    
    # Вычисляем точность классификации
    accuracy = compute_per_point_accuracy(class_logits, class_targets)
    
    print(f"Размер входа: {x.shape}")
    print(f"Размер маски классов: {class_mask.shape}")
    print(f"Размер реконструкции кривых: {reconstructed_curves.shape}")
    print(f"Размер латентного представления: {mu.shape}")
    print(f"Размер логитов класса: {class_logits.shape}")
    print(f"Размер вероятностей класса: {class_probs.shape}")
    print(f"Размер целевых меток класса: {class_targets.shape}")
    print(f"Общая потеря: {total_loss.item():.4f}")
    print(f"Потеря восстановления: {recon_loss.item():.4f}")
    print(f"KL-дивергенция: {kl_loss.item():.4f}")
    print(f"Потеря классификации: {class_loss.item():.4f}")
    print(f"Точность классификации: {accuracy.item():.4f}")


if __name__ == "__main__":
    example_usage()
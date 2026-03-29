import torch
import torch.nn as nn
import torch.nn.functional as F


def vae_multitask_loss(curve_true, label_true, curve_pred, label_pred, mu, logvar, 
                      curve_weight=1.0, label_weight=1.0, kld_weight=1.0):
    """
    Функция потерь для многозадачного VAE, который восстанавливает как кривую, так и метки точек
    
    Args:
        curve_true: Истинные значения кривой (batch, 3, seq_len)
        label_true: Истинные метки точек (batch, 1, seq_len) или (batch, seq_len)
        curve_pred: Предсказанные значения кривой (batch, 3, seq_len)
        label_pred: Предсказанные метки точек (batch, 1, seq_len)
        mu: Среднее значение в латентном пространстве (batch, latent_dim)
        logvar: Логарифм дисперсии в латентном пространстве (batch, latent_dim)
        curve_weight: Вес для потерь кривой
        label_weight: Вес для потерь меток
        kld_weight: Вес для KL-дивергенции
    """
    
    # Потеря восстановления кривой (MSE)
    curve_loss = F.mse_loss(curve_pred, curve_true)
    
    # Потеря для предсказания меток (Binary Cross Entropy)
    # Убедимся, что label_true имеет правильную форму
    if len(label_true.shape) == 2:
        label_true = label_true.unsqueeze(1)  # (batch, 1, seq_len)
    
    label_loss = F.binary_cross_entropy(label_pred, label_true.float())
    
    # KL-дивергенция
    kld_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    
    # Общая потеря
    total_loss = curve_weight * curve_loss + label_weight * label_loss + kld_weight * kld_loss
    
    return total_loss, curve_loss, label_loss, kld_loss


def get_accuracy(label_true, label_pred, threshold=0.5):
    """
    Вычисляет точность для задачи бинарной классификации меток точек
    
    Args:
        label_true: Истинные метки (batch, seq_len) или (batch, 1, seq_len)
        label_pred: Предсказанные вероятности (batch, 1, seq_len)
        threshold: Порог для бинаризации предсказаний
    """
    if len(label_true.shape) == 2:
        label_true = label_true.unsqueeze(1)  # (batch, 1, seq_len)
    
    predicted_labels = (label_pred >= threshold).float()
    accuracy = (predicted_labels == label_true).float().mean()
    
    return accuracy.item()
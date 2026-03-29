import torch
import torch.nn.functional as F

def vae_loss(recon_x, x, mu, logvar, recon_weight=1.0, kl_weight=1.0):
    # MEAN reduction делает лосс независимым от размера батча
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    
    # KL тоже усредняем по батчу и латентным размерностям
    # -0.5 * sum(1 + logvar - mu^2 - exp(logvar)) / batch_size
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    total_loss = recon_weight * recon_loss + kl_weight * kl_loss
    return total_loss, recon_loss, kl_loss
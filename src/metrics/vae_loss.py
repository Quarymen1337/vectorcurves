import torch
import torch.nn.functional as F

def vae_loss(recon_x, x, mu, logvar, recon_weight=1.0, kl_weight=1.0):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_weight * recon_loss + kl_weight * kl_loss
    return total_loss, recon_loss, kl_loss
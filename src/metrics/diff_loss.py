import torch.nn.functional as F

def diff_loss(reconstructed, target, beta = 0):
    mse_loss = F.mse_loss(reconstructed, target)
    v_reconstructed = reconstructed[:, 1:, :] - reconstructed[:, :-1, :]
    v_target = target[:, 1:, :] - target[:, :-1, :]
    velocity_loss = F.mse_loss(v_reconstructed, v_target)
    
    return mse_loss + beta * velocity_loss
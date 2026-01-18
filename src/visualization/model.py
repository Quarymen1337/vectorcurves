import torch
import numpy as np
import matplotlib.pyplot as plt
def visualize_reconstruction_vae(model, val_dataset, num_examples=5, device='cuda'):
    """
    Визуализация реконструкции для VAE модели
    """
    model.eval()
    
    indices = np.random.choice(len(val_dataset), num_examples, replace=False)
    fig, axes = plt.subplots(num_examples, 3, figsize=(18, 4*num_examples))
    
    if num_examples == 1:
        axes = axes.reshape(1, -1)
    
    total_mse = 0
    with torch.no_grad():
        for i, idx in enumerate(indices):
            signal_norm, signal_orig = val_dataset[idx]  
            signal_input = signal_norm.unsqueeze(0).to(device)  
            
            reconstructed, mu, logvar, z = model(signal_input)
            
            signal_norm = signal_norm.cpu().numpy()  
            reconstructed = reconstructed.squeeze(0).cpu().numpy() 
            
            example_mse = 0
            for ch in range(3):
                ax = axes[i, ch]
                
                ax.plot(signal_norm[:, ch], 'b-', linewidth=2, label='Original', alpha=0.8)
                ax.plot(reconstructed[:, ch], 'r--', linewidth=2, label='Reconstructed', alpha=0.8)
                
                mse_ch = np.mean((signal_norm[:, ch] - reconstructed[:, ch])**2)
                example_mse += mse_ch
                
                ax.set_title(f'Example {i+1}, Channel {ch+1}\nMSE: {mse_ch:.6f}')
                ax.set_xlabel('Time steps')
                ax.set_ylabel('Amplitude')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            example_mse /= 3  
            total_mse += example_mse
            
            latent_info = f'Index: {idx}\nLatent dim: {z.shape[1]}\nZ norm: {torch.norm(z).item():.3f}'
            axes[i, 0].text(0.02, 0.98, latent_info, transform=axes[i, 0].transAxes, 
                           verticalalignment='top', fontsize=8,
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    avg_mse = total_mse / num_examples
    plt.tight_layout()
    plt.suptitle(f'VAE Signal Reconstruction - Average MSE: {avg_mse:.6f}', y=1.02, fontsize=16)
    plt.savefig('vae_reconstruction_validation.png', bbox_inches='tight', dpi=300)
    plt.show()
    
    return avg_mse
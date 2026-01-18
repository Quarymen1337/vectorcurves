from src.data.loader import process_protonated_files, create_data
from src.preprocessing.interpolation import resample_trajectory
from src.preprocessing.atom_process import get_ca_coordinates
from src.preprocessing.curves import get_invariant_features

from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from src.data.dataset import Trajectories_dataset
from src.models.vae_v2 import TCNVAE
from src.metrics.vae_loss import vae_loss
import torch.optim as optim
import torch
import torch.nn.functional as F
import warnings
import os

warnings.filterwarnings('ignore')

from src.utils.config_loader import project_config
import torch



config = project_config.load()  # загружает configs/config.yaml
print(f"\n Основная информация:")
print(f"   Проект: {config.get('project', {}).get('name', 'N/A')}")
print(f"   Версия: {config.get('project', {}).get('version', 'N/A')}")
print(f"   Автор: {config.get('project', {}).get('author', 'N/A')}")
print(f"   Будет обработано: {config.get('dataset', {}).get('length', 1000)} примеров")
print(f"   Обучение будет происходить: {config.get('model', {}).get('device', 'cpu')}")
print(f"   Размеры примеров(в атомах): от {config.get('dataset', {}).get('min_atom_len', 1)} до {config.get('dataset', {}).get('max_atom_len', 1000)} ")







JSON_DATA = process_protonated_files('data/by_pairs_dataset', max_l = config.get('dataset', {}).get('length', 1000), alpha=config.get('dataset', {}).get('min_atom_len', 1), beta=config.get('dataset', {}).get('max_atom_len', 1000))
trace_data, graph_data, ca_coordinate_data = create_data(JSON_DATA, K = 1024)
print(ca_coordinate_data[0])
train_data_raw, val_data_raw = train_test_split(
    trace_data,       
    test_size=config.get('training', {}).get('validation_split', 0.1),  
    random_state=42 
)


device = config.get('model', {}).get('cuda', 'cpu')
learning_rate = config.get('training', {}).get('learning_rate', 0.1)
num_epochs = config.get('training', {}).get('epochs', 100)
batch_size = config.get('training', {}).get('batch_size', 16)

#####

kl_weight_start = 0.05      
kl_weight_end = 0.05
kl_anneal_epochs = 1  

#####


train_dataset = Trajectories_dataset(train_data_raw, normalize=True)
val_dataset = Trajectories_dataset(val_data_raw, normalize=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
model = TCNVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
     

train_losses = []
val_losses = []
train_recon_losses = []
val_recon_losses = []
train_kl_losses = []
val_kl_losses = []
train_mse_losses = []
val_mse_losses = []

print('Начало обучения VAE с KL Annealing')
for epoch in range(num_epochs):
    if epoch < kl_anneal_epochs:
        kl_weight = kl_weight_start + (kl_weight_end - kl_weight_start) * (epoch / kl_anneal_epochs)
    else:
        kl_weight = kl_weight_end

    model.train()
    train_total_loss = 0
    train_recon_loss = 0
    train_kl_loss = 0
    train_mse = 0
    
    for batch in train_loader:
        signals = batch[0].to(device) 
        
        reconstructed, mu, logvar, z = model(signals)
        
        total_loss, recon_loss, kl_loss = vae_loss(
            reconstructed, signals, mu, logvar, 
            recon_weight=1.0, kl_weight=kl_weight
        )
        
        mse_loss = F.mse_loss(reconstructed, signals, reduction='mean')
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        train_total_loss += total_loss.item()
        train_recon_loss += recon_loss.item()
        train_kl_loss += kl_loss.item()
        train_mse += mse_loss.item()
    
    model.eval()
    val_total_loss = 0
    val_recon_loss = 0
    val_kl_loss = 0
    val_mse = 0
    
    with torch.no_grad():
        for batch in val_loader:
            signals = batch[0].to(device)  
            reconstructed, mu, logvar, z = model(signals)

            total_loss, recon_loss, kl_loss = vae_loss(
                reconstructed, signals, mu, logvar,
                recon_weight=1.0, kl_weight=kl_weight
            )
            
            mse_loss = F.mse_loss(reconstructed, signals, reduction='mean')
            
            val_total_loss += total_loss.item()
            val_recon_loss += recon_loss.item()
            val_kl_loss += kl_loss.item()
            val_mse += mse_loss.item()
    
    # Усреднение лоссов
    avg_train_total = train_total_loss / len(train_loader)
    avg_train_recon = train_recon_loss / len(train_loader)
    avg_train_kl = train_kl_loss / len(train_loader)
    avg_train_mse = train_mse / len(train_loader)
    
    avg_val_total = val_total_loss / len(val_loader)
    avg_val_recon = val_recon_loss / len(val_loader)
    avg_val_kl = val_kl_loss / len(val_loader)
    avg_val_mse = val_mse / len(val_loader)
        
    if ((epoch + 1) % 10 == 0) or (epoch == 0):
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'  Train - Total: {avg_train_total:.6f} | Recon: {avg_train_recon:.6f} | KL (unweighted): {avg_train_kl:.6f} | MSE: {avg_train_mse:.6f}')
        print(f'  Val   - Total: {avg_val_total:.6f} | Recon: {avg_val_recon:.6f} | KL (unweighted): {avg_val_kl:.6f} | MSE: {avg_val_mse:.6f}')
        print(f'  Current KL Weight: {kl_weight:.6f}') # Вывод текущего веса
        print('-' * 80)
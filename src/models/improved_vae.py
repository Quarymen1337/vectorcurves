import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.activation = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        residual = x
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.activation(out)
        return out


class ImprovedConv1DVAE(nn.Module):
    def __init__(self, latent_dim=128, input_channels=4, sequence_length=2048):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.input_channels = input_channels  # Updated to 4 channels (3 for curve + 1 for labels)
        
        # Энкодер - теперь принимает 4 канала
        self.encoder = nn.Sequential(
            # Input: (batch, 4, 2048)
            nn.Conv1d(input_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            
            ResidualBlock(32),
            
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            
            ResidualBlock(64),
            
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            
            ResidualBlock(128),
            
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            
            ResidualBlock(256),
            
            nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            
            ResidualBlock(512),
            
            nn.AdaptiveAvgPool1d(16)  
        )
        
        self.fc_mu = nn.Linear(512 * 16, latent_dim)
        self.fc_logvar = nn.Linear(512 * 16, latent_dim)
        
        self.fc_decode = nn.Linear(latent_dim, 512 * 16)
        
        # Основной декодер для восстановления первых 3 каналов (кривой)
        self.curve_decoder = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            
            ResidualBlock(512),
            
            nn.Upsample(scale_factor=2, mode='linear'),  
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            
            ResidualBlock(256),
            
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            
            nn.Upsample(scale_factor=2, mode='linear'),  
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            
            ResidualBlock(128),
            
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            
            nn.Upsample(scale_factor=2, mode='linear'),  
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            
            ResidualBlock(64),
            
            nn.Upsample(scale_factor=2, mode='linear'), 
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            
            ResidualBlock(32),
            
            nn.Upsample(scale_factor=2, mode='linear'), 
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            
            ResidualBlock(16),
            
            nn.Upsample(scale_factor=4, mode='linear'),  
            nn.Conv1d(16, 3, kernel_size=3, padding=1),  # Output only 3 channels for curve
            nn.Sigmoid()  
        )
        
        # Декодер для предсказания меток (4-й канал)
        self.label_decoder = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            
            ResidualBlock(256),
            
            nn.Upsample(scale_factor=2, mode='linear'),  
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            
            ResidualBlock(128),
            
            nn.Upsample(scale_factor=2, mode='linear'),  
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            
            ResidualBlock(64),
            
            nn.Upsample(scale_factor=2, mode='linear'), 
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            
            ResidualBlock(32),
            
            nn.Upsample(scale_factor=2, mode='linear'), 
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            
            ResidualBlock(16),
            
            nn.Upsample(scale_factor=4, mode='linear'),  
            nn.Conv1d(16, 1, kernel_size=3, padding=1),  # Output 1 channel for binary labels
            nn.Sigmoid()  # Sigmoid activation for binary classification
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1) 
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_curve(self, z):
        h = self.fc_decode(z)
        h = h.view(h.size(0), 512, 16)  
        return self.curve_decoder(h) 

    def decode_labels(self, z):
        h = self.fc_decode(z)
        h = h.view(h.size(0), 512, 16)  
        return self.label_decoder(h) 

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        curve_recon = self.decode_curve(z)
        label_recon = self.decode_labels(z)
        return curve_recon, label_recon, mu, logvar
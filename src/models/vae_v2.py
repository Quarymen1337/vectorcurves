import torch
import torch.nn as nn
import torch.nn.functional as F

class TCNResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.1):
        super(TCNResidualBlock, self).__init__()
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                               padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)

class TCNVAE(nn.Module):
    def __init__(self, latent_dim=256, input_channels=3):
        super(TCNVAE, self).__init__()
        
        self.latent_dim = latent_dim
        
        # --- ENCODER ---
        # Вход: (B, 3, 1024)
        self.enc_layers = nn.ModuleList([
            TCNResidualBlock(input_channels, 32, kernel_size=7, dilation=1),
            nn.MaxPool1d(2), # -> 512
            TCNResidualBlock(32, 64, kernel_size=7, dilation=2),
            nn.MaxPool1d(2), # -> 256
            TCNResidualBlock(64, 128, kernel_size=5, dilation=4),
            nn.MaxPool1d(2), # -> 128
            TCNResidualBlock(128, 256, kernel_size=5, dilation=8),
            nn.MaxPool1d(2), # -> 64
            TCNResidualBlock(256, 256, kernel_size=3, dilation=16),
            nn.MaxPool1d(2)  # -> 32
        ])
        
        # Используем Global Average Pooling, чтобы уменьшить число параметров
        self.gap = nn.AdaptiveAvgPool1d(1) 
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # --- DECODER ---
        # Начинаем с латентного вектора и разворачиваем его в последовательность длиной 32
        self.dec_fc = nn.Linear(latent_dim, 256)
        
        self.dec_layers = nn.ModuleList([
            TCNResidualBlock(256, 256, kernel_size=3, dilation=16),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True), # -> 64
            TCNResidualBlock(256, 128, kernel_size=5, dilation=8),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True), # -> 128
            TCNResidualBlock(128, 64, kernel_size=5, dilation=4),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True), # -> 256
            TCNResidualBlock(64, 32, kernel_size=7, dilation=2),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True), # -> 512
            TCNResidualBlock(32, 32, kernel_size=7, dilation=1),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True)  # -> 1024
        ])
        
        self.final_conv = nn.Conv1d(32, input_channels, kernel_size=1)

    def encode(self, x):
        for layer in self.enc_layers:
            x = layer(x)
        x = self.gap(x).view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.dec_fc(z)
        x = x.unsqueeze(-1).repeat(1, 1, 32) # (B, 256, 32)
        
        for layer in self.dec_layers:
            x = layer(x)
        return self.final_conv(x)

    def forward(self, x):
        # x: (B, 1024, 3) -> (B, 3, 1024)
        x = x.transpose(1, 2)
        
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        
        # recon: (B, 3, 1024) -> (B, 1024, 3)
        return recon.transpose(1, 2), mu, logvar, z
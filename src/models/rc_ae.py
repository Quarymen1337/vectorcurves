import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation, kernel_size=3):
        super(ResidualBlock, self).__init__()
        padding = dilation * (kernel_size - 1) // 2
        
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, 
                      padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels)
        )
        self.relu = nn.ReLU(True)

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out += residual
        return self.relu(out)

class ResConvModel(nn.Module):
    def __init__(self, input_channels=3, sequence_length=1024, latent_dim=256, dropout_rate=0.05):
        super(ResConvModel, self).__init__()
        
        self.initial_conv = nn.Conv1d(input_channels, 32, kernel_size=7, padding=3)
        
        self.enc_layers = nn.ModuleList([
            ResidualBlock(32, dilation=1),
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
            
            ResidualBlock(64, dilation=2),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1), 
            
            ResidualBlock(128, dilation=4),
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1), 
            
            ResidualBlock(256, dilation=8),
            nn.Conv1d(256, 512, kernel_size=4, stride=2, padding=1), 
            
            ResidualBlock(512, dilation=16),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1),
        ])
        
        self.final_length = sequence_length // (2**5) 
        self.encoded_size = 512 * self.final_length 
        
        self.latent_layer = nn.Linear(self.encoded_size, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, self.encoded_size)

        self.dec_layers = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True), 
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            ResidualBlock(512, dilation=8),
            
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True), 
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            ResidualBlock(256, dilation=4),
            
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True), 
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            ResidualBlock(128, dilation=2),
            
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True), 
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            ResidualBlock(64, dilation=1),
            
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
        ])
        self.final_out = nn.Sequential(
            nn.Conv1d(32, input_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.initial_conv(x)
        for layer in self.enc_layers:
            x = layer(x)
        x = x.view(x.size(0), -1)
        return self.latent_layer(x)

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(x.size(0), 512, self.final_length)
        for layer in self.dec_layers:
            x = layer(x)
        return self.final_out(x)

    def forward(self, x):
        x = x.transpose(1, 2)
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed.transpose(1, 2), latent
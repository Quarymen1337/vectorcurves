import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvAutoencoder(nn.Module):
    def __init__(self, input_channels=3, sequence_length=1024, latent_dim=256, dropout_rate=0.05):
        super(ConvAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=5, stride=2, padding=2),  
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2), 
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),  
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2), 
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            
            nn.Conv1d(256, 512, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
        )
        
        self.final_length = sequence_length // (2**5) 
        self.encoded_size = 512 * self.final_length 
        
        self.latent_layer = nn.Linear(self.encoded_size, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, self.encoded_size)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(512, 256, kernel_size=5, stride=2, padding=2, output_padding=1), 
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            
            nn.ConvTranspose1d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1), 
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1), 
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1), 
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            
            nn.ConvTranspose1d(32, input_channels, kernel_size=5, stride=2, padding=2, output_padding=1), 
            nn.Sigmoid()  # Выход в диапазоне [0, 1]
        )
    
    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        latent = self.latent_layer(x)
        return latent
    
    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(x.size(0), 512, self.final_length)
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        x = x.transpose(1, 2)
        
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        
        reconstructed = reconstructed.transpose(1, 2)
        
        return reconstructed, latent
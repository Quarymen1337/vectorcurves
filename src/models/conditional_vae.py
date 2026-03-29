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


class ConditionalImprovedConv1DVAE(nn.Module):
    def __init__(self, latent_dim=128, input_channels=3, sequence_length=2048, condition_dim=2):
        super().__init__()

        self.latent_dim = latent_dim
        self.condition_dim = condition_dim

        # Энкодер
        self.encoder = nn.Sequential(
            # Input: (batch, 3, 2048)
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

        self.fc_mu = nn.Linear(512 * 16 + condition_dim, latent_dim)
        self.fc_logvar = nn.Linear(512 * 16 + condition_dim, latent_dim)

        self.fc_decode_condition = nn.Linear(condition_dim, condition_dim)
        self.fc_decode = nn.Linear(latent_dim + condition_dim, 512 * 16)

        self.decoder = nn.Sequential(
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
            nn.Conv1d(16, input_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x, conditions):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        
        # Конкатенируем условия с признаками энкодера
        h_with_conditions = torch.cat([h, conditions], dim=1)
        
        mu = self.fc_mu(h_with_conditions)
        logvar = self.fc_logvar(h_with_conditions)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, conditions):
        # Обрабатываем условия отдельно
        processed_conditions = self.fc_decode_condition(conditions)
        
        # Конкатенируем латентное представление с обработанными условиями
        z_with_conditions = torch.cat([z, processed_conditions], dim=1)
        
        h = self.fc_decode(z_with_conditions)
        h = h.view(h.size(0), 512, 16)
        return self.decoder(h)

    def forward(self, x, conditions):
        mu, logvar = self.encode(x, conditions)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, conditions)
        return recon_x, mu, logvar

    def sample(self, num_samples, conditions, device='cpu'):
        """
        Генерирует новые образцы на основе условий
        :param num_samples: количество образцов для генерации
        :param conditions: тензор условий формы (num_samples, condition_dim)
        :param device: устройство для вычислений
        """
        with torch.no_grad():
            # Генерируем случайные латентные векторы
            z = torch.randn(num_samples, self.latent_dim).to(device)
            
            # Декодируем с учетом условий
            samples = self.decode(z, conditions)
            return samples
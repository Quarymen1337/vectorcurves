"""
Тесты для моделей
"""

import pytest
import torch
from src.models.rc_ae import ResConvModel
from src.models.vae import ConvVAE


class TestResConvModel:
    """Тесты для ResConvModel"""

    def test_init(self):
        """Инициализация модели"""
        model = ResConvModel(sequence_length=128, latent_dim=32)
        assert model is not None

    def test_forward(self):
        """Прямой проход"""
        model = ResConvModel(sequence_length=128, latent_dim=32)
        x = torch.randn(2, 128, 3)
        reconstructed, latent = model(x)
        assert reconstructed.shape == (2, 128, 3)
        assert latent.shape == (2, 32)

    def test_encode(self):
        """Кодирование"""
        model = ResConvModel(sequence_length=128, latent_dim=32)
        x = torch.randn(2, 3, 128)
        latent = model.encode(x)
        assert latent.shape == (2, 32)

    def test_decode(self):
        """Декодирование"""
        model = ResConvModel(sequence_length=128, latent_dim=32)
        z = torch.randn(2, 32)
        reconstructed = model.decode(z)
        assert reconstructed.shape == (2, 3, 128)


class TestConvVAE:
    """Тесты для ConvVAE"""

    def test_init(self):
        """Инициализация VAE"""
        model = ConvVAE(sequence_length=128, latent_dim=32)
        assert model is not None

    def test_forward(self):
        """Прямой проход VAE"""
        model = ConvVAE(sequence_length=128, latent_dim=32)
        x = torch.randn(2, 128, 3)
        reconstructed, mu, logvar, z = model(x)
        assert reconstructed.shape == (2, 128, 3)
        assert mu.shape == (2, 32)
        assert logvar.shape == (2, 32)
        assert z.shape == (2, 32)

    def test_reparameterize(self):
        """Репараметризация"""
        model = ConvVAE(sequence_length=128, latent_dim=32)
        mu = torch.randn(2, 32)
        logvar = torch.randn(2, 32)
        z = model.reparameterize(mu, logvar)
        assert z.shape == (2, 32)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

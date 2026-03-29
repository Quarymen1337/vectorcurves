"""
Классы моделей: VAE, AE и их вариации
"""

from .vae import ConvVAE
from .rc_ae import ResConvModel

__all__ = ['ConvVAE', 'ResConvModel']

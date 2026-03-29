"""
Анализ результатов обучения и латентного пространства
"""

from .analysis_tools import (
    analyze_training_history,
    evaluate_model_quality,
    analyze_latent_space,
    generate_and_visualize
)

from .latent_analysis import (
    analyze_latent_space_statistics,
    check_posterior_collapse
)

from .tsne_analysis import (
    visualize_latent_space_tsne_1,
    analyze_latent_space_detailed,
    plot_multiple_perplexities
)

__all__ = [
    'analyze_training_history',
    'evaluate_model_quality',
    'analyze_latent_space',
    'generate_and_visualize',
    'analyze_latent_space_statistics',
    'check_posterior_collapse',
    'visualize_latent_space_tsne_1',
    'analyze_latent_space_detailed',
    'plot_multiple_perplexities'
]

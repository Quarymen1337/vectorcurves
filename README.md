# VectorCurves

Проект для генерации и анализа структур белков на основе кривых с использованием вариационных автоэнкодеров (VAE).

## Структура проекта

```
vectorcurves/
├── src/                        # Исходный код проекта
│   ├── analysis/               # Анализ результатов и визуализация
│   │   ├── analysis_tools.py
│   │   ├── latent_analysis.py
│   │   └── tsne_analysis.py
│   ├── data/                   # Загрузка и обработка данных
│   │   ├── dataset.py
│   │   └── loader.py
│   ├── features/               # Извлечение признаков
│   ├── metrics/                # Функции потерь и метрики
│   ├── models/                 # Архитектуры моделей (VAE, AE)
│   ├── pdb/                    # Обработка PDB файлов
│   ├── preprocessing/          # Предобработка данных
│   ├── utils/                  # Утилиты
│   └── visualization/          # Визуализация результатов
├── tests/                      # Тесты
├── notebooks/                  # Jupyter ноутбуки с экспериментами
├── configs/                    # Конфигурационные файлы
├── models/                     # Сохраненные модели
├── data/                       # Данные для обучения
├── train_vae.py                # Основной скрипт обучения
├── analyze_model.py            # Скрипт анализа модели
├── RUN_AE.py                   # Скрипт обучения AE (устаревший)
└── requirements.txt            # Зависимости
```

## Установка

### Требования

- Python 3.11+
- CUDA (опционально, для GPU)

### Установка зависимостей

```bash
# Создание виртуального окружения
python3 -m venv venv

# Активация окружения
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows

# Установка зависимостей
pip install -r requirements.txt
```

## Быстрый старт

### 1. Настройка конфигурации

Отредактируйте `configs/config.yaml`:

```yaml
dataset:
  path: 'data/dompdb'  # Путь к вашим PDB файлам
  spline_interpolation_points: 512  # Количество точек ресемплинга

model:
  device: 'cuda'  # 'cuda' или 'cpu'
  latent_dim: 64  # Размерность латентного пространства

training:
  epochs: 200
  batch_size: 32
  learning_rate: 0.001
  early_stoping: 15
  validation_split: 0.1
```

### 2. Обучение модели

```bash
# Обучение VAE
python train_vae.py

# Обучение с конкретной конфигурацией
python train_vae.py --config configs/config.yaml
```

### 3. Анализ модели

```bash
# Анализ обученной модели
python analyze_model.py --model best_vae_model.pth

# Анализ латентного пространства
python analyze_model.py --model best_vae_model.pth --analyze-latent

# Построение графика истории обучения
python analyze_model.py --model best_vae_model.pth --plot-history
```

## Использование в коде

### Загрузка данных

```python
from src.data.loader import load_and_preprocess_pdb_files
from src.preprocessing.resample import resample_trajectories

# Загрузка PDB файлов
pdb_files, ca_points = load_and_preprocess_pdb_files('data/dompdb')

# Ресемплинг кривых
resampled = resample_trajectories(ca_points, k_points=512)
```

### Создание модели

```python
from src.models.vae import ConvVAE
from src.models.rc_ae import ResConvModel

# VAE модель
vae = ConvVAE(
    input_channels=3,
    sequence_length=512,
    latent_dim=64
)

# AE модель
ae = ResConvModel(
    input_channels=3,
    sequence_length=512,
    latent_dim=64
)
```

### Обучение

```python
from torch.utils.data import DataLoader
from src.data.dataset import TrajectoriesDataset
import torch.optim as optim

# Создание датасета
dataset = TrajectoriesDataset(resampled, normalize=True)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Обучение
optimizer = optim.Adam(vae.parameters(), lr=0.001)

for epoch in range(100):
    for batch in dataloader:
        x, _ = batch
        reconstructed, mu, logvar, z = vae(x)
        # Вычисление loss и оптимизация
```

### Анализ латентного пространства

```python
from src.analysis import (
    analyze_latent_space_statistics,
    check_posterior_collapse,
    visualize_latent_space_tsne_1
)

# Статистики латентного пространства
stats = analyze_latent_space_statistics(vae, dataloader, device)

# Проверка posterior collapse
collapse_info = check_posterior_collapse(vae, dataloader, device)

# t-SNE визуализация
tsne_result = visualize_latent_space_tsne_1(vae, dataloader, device)
```

## Запуск тестов

```bash
# Запуск всех тестов
pytest tests/ -v

# Запуск конкретного теста
pytest tests/test_models.py -v
```

## Ноутбуки

В папке `notebooks/` находятся примеры использования:

- `FixedNewDataVae.ipynb` - основной ноутбук с VAE (рекомендуется)
- `Analys.ipynb` - анализ данных
- `CurveToAtom.ipynb` - конвертация кривых в атомы

## Конфигурация

Параметры в `configs/config.yaml`:

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `dataset.path` | Путь к PDB файлам | `data/dompdb` |
| `dataset.spline_interpolation_points` | Точек ресемплинга | `1024` |
| `model.device` | Устройство (cuda/cpu) | `cuda` |
| `model.latent_dim` | Размерность латентного пространства | `64` |
| `training.epochs` | Количество эпох | `200` |
| `training.batch_size` | Размер батча | `32` |
| `training.learning_rate` | Скорость обучения | `0.001` |
| `training.early_stoping` | Патинс для ранней остановки | `15` |
| `training.validation_split` | Доля валидации | `0.1` |

## Авторы

- **Polyakov Stepan** - *Initial work*

## Лицензия

MIT License

## Благодарности

Проект разработан для анализа и генерации структур белков с использованием методов глубокого обучения.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def compare_curves_3d(original_curve, reconstructed_curve=None, title="Сравнение кривых в 3D", 
                     labels=("Оригинал", "Реконструкция"), colors=('blue', 'red')):
    """
    Функция для визуализации и сравнения кривых в 3D пространстве
    
    Параметры:
    - original_curve: numpy array или torch tensor формы (sequence_length, 3) - оригинальная кривая
    - reconstructed_curve: numpy array или torch tensor формы (sequence_length, 3) - реконструированная кривая (опционально)
    - title: строка - заголовок графика
    - labels: кортеж строк - метки для оригинала и реконструкции
    - colors: кортеж цветов - цвета для оригинала и реконструкции
    
    Возвращает:
    - fig, ax: объекты matplotlib для дальнейшей настройки
    """
    # Преобразование в numpy массив, если это torch tensor
    if hasattr(original_curve, 'cpu'):
        original_curve = original_curve.cpu().numpy()
    elif hasattr(original_curve, 'detach'):
        original_curve = original_curve.detach().cpu().numpy()
    
    if reconstructed_curve is not None:
        if hasattr(reconstructed_curve, 'cpu'):
            reconstructed_curve = reconstructed_curve.cpu().numpy()
        elif hasattr(reconstructed_curve, 'detach'):
            reconstructed_curve = reconstructed_curve.detach().cpu().numpy()
    
    # Проверка формата данных
    if original_curve.ndim == 2 and original_curve.shape[1] == 3:
        # Формат (sequence_length, 3) - X, Y, Z координаты
        x_orig, y_orig, z_orig = original_curve[:, 0], original_curve[:, 1], original_curve[:, 2]
    elif original_curve.ndim == 3 and original_curve.shape[0] == 3:
        # Формат (3, sequence_length) - X, Y, Z координаты
        x_orig, y_orig, z_orig = original_curve[0, :], original_curve[1, :], original_curve[2, :]
    else:
        raise ValueError(f"Неправильный формат данных для original_curve: {original_curve.shape}. Ожидается (N, 3) или (3, N)")
    
    # Создание 3D графика
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Построение оригинальной кривой
    ax.plot(x_orig, y_orig, z_orig, label=labels[0], color=colors[0], linewidth=2, alpha=0.8)
    
    # Построение реконструированной кривой, если она предоставлена
    if reconstructed_curve is not None:
        if reconstructed_curve.ndim == 2 and reconstructed_curve.shape[1] == 3:
            x_rec, y_rec, z_rec = reconstructed_curve[:, 0], reconstructed_curve[:, 1], reconstructed_curve[:, 2]
        elif reconstructed_curve.ndim == 3 and reconstructed_curve.shape[0] == 3:
            x_rec, y_rec, z_rec = reconstructed_curve[0, :], reconstructed_curve[1, :], reconstructed_curve[2, :]
        else:
            raise ValueError(f"Неправильный формат данных для reconstructed_curve: {reconstructed_curve.shape}. Ожидается (N, 3) или (3, N)")
        
        ax.plot(x_rec, y_rec, z_rec, label=labels[1], color=colors[1], linewidth=2, alpha=0.8)
    
    # Настройка графика
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    
    # Улучшение визуализации
    ax.grid(True)
    
    plt.tight_layout()
    return fig, ax


def compare_multiple_curves_3d(curve_pairs, titles=None, labels=("Оригинал", "Реконструкция"), 
                              colors=('blue', 'red'), cols=2):
    """
    Функция для визуализации нескольких пар кривых в 3D
    
    Параметры:
    - curve_pairs: список пар (оригинал, реконструкция)
    - titles: список заголовков для каждого графика
    - labels: кортеж меток
    - colors: кортеж цветов
    - cols: количество столбцов в сетке графиков
    """
    n_pairs = len(curve_pairs)
    rows = int(np.ceil(n_pairs / cols))
    
    fig = plt.figure(figsize=(cols * 6, rows * 6))
    
    for i, (orig, rec) in enumerate(curve_pairs):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        
        # Обработка оригинальной кривой
        if hasattr(orig, 'cpu'):
            orig = orig.cpu().numpy()
        elif hasattr(orig, 'detach'):
            orig = orig.detach().cpu().numpy()
        
        if orig.ndim == 2 and orig.shape[1] == 3:
            x_orig, y_orig, z_orig = orig[:, 0], orig[:, 1], orig[:, 2]
        elif orig.ndim == 3 and orig.shape[0] == 3:
            x_orig, y_orig, z_orig = orig[0, :], orig[1, :], orig[2, :]
        else:
            raise ValueError(f"Неправильный формат данных для оригинальной кривой: {orig.shape}")
        
        # Обработка реконструированной кривой
        if hasattr(rec, 'cpu'):
            rec = rec.cpu().numpy()
        elif hasattr(rec, 'detach'):
            rec = rec.detach().cpu().numpy()
        
        if rec.ndim == 2 and rec.shape[1] == 3:
            x_rec, y_rec, z_rec = rec[:, 0], rec[:, 1], rec[:, 2]
        elif rec.ndim == 3 and rec.shape[0] == 3:
            x_rec, y_rec, z_rec = rec[0, :], rec[1, :], rec[2, :]
        else:
            raise ValueError(f"Неправильный формат данных для реконструированной кривой: {rec.shape}")
        
        # Построение кривых
        ax.plot(x_orig, y_orig, z_orig, label=labels[0], color=colors[0], linewidth=2, alpha=0.8)
        ax.plot(x_rec, y_rec, z_rec, label=labels[1], color=colors[1], linewidth=2, alpha=0.8)
        
        # Настройка графика
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        if titles and i < len(titles):
            ax.set_title(titles[i])
        else:
            ax.set_title(f'Пара {i+1}')
        
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    return fig


def calculate_reconstruction_metrics(original_curve, reconstructed_curve):
    """
    Функция для вычисления метрик качества реконструкции
    
    Параметры:
    - original_curve: numpy array или torch tensor формы (sequence_length, 3) - оригинальная кривая
    - reconstructed_curve: numpy array или torch tensor формы (sequence_length, 3) - реконструированная кривая
    
    Возвращает:
    - словарь с метриками: MSE, MAE, RMSE
    """
    # Преобразование в numpy массив
    if hasattr(original_curve, 'cpu'):
        original_curve = original_curve.cpu().numpy()
    elif hasattr(original_curve, 'detach'):
        original_curve = original_curve.detach().cpu().numpy()
    
    if hasattr(reconstructed_curve, 'cpu'):
        reconstructed_curve = reconstructed_curve.cpu().numpy()
    elif hasattr(reconstructed_curve, 'detach'):
        reconstructed_curve = reconstructed_curve.detach().cpu().numpy()
    
    # Проверка формата данных
    if original_curve.ndim == 2 and original_curve.shape[1] == 3:
        pass  # Формат (sequence_length, 3) уже правильный
    elif original_curve.ndim == 3 and original_curve.shape[0] == 3:
        original_curve = original_curve.T  # Преобразование в (sequence_length, 3)
    
    if reconstructed_curve.ndim == 2 and reconstructed_curve.shape[1] == 3:
        pass  # Формат (sequence_length, 3) уже правильный
    elif reconstructed_curve.ndim == 3 and reconstructed_curve.shape[0] == 3:
        reconstructed_curve = reconstructed_curve.T  # Преобразование в (sequence_length, 3)
    
    # Проверка совпадения размерностей
    if original_curve.shape != reconstructed_curve.shape:
        raise ValueError(f"Размерности оригинальной и реконструированной кривых не совпадают: "
                         f"{original_curve.shape} vs {reconstructed_curve.shape}")
    
    # Вычисление метрик
    mse = np.mean((original_curve - reconstructed_curve) ** 2)
    mae = np.mean(np.abs(original_curve - reconstructed_curve))
    rmse = np.sqrt(mse)
    
    # Вычисление среднего евклидова расстояния между соответствующими точками
    euclidean_distances = np.sqrt(np.sum((original_curve - reconstructed_curve) ** 2, axis=1))
    mean_euclidean_distance = np.mean(euclidean_distances)
    
    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'Mean_Euclidean_Distance': mean_euclidean_distance
    }


# Пример использования функций
if __name__ == "__main__":
    # Создание примера данных
    t = np.linspace(0, 4*np.pi, 2048)
    original_curve = np.column_stack([
        np.sin(t),           # X
        np.cos(t),           # Y
        0.5 * t              # Z
    ])
    
    # Создание немного измененной "реконструированной" кривой
    reconstructed_curve = np.column_stack([
        0.95 * np.sin(t) + 0.01 * np.random.normal(0, 1, len(t)),  # X
        0.98 * np.cos(t) + 0.01 * np.random.normal(0, 1, len(t)),  # Y
        0.48 * t + 0.01 * np.random.normal(0, 1, len(t))          # Z
    ])
    
    # Визуализация
    fig, ax = compare_curves_3d(
        original_curve, 
        reconstructed_curve, 
        title="Пример сравнения кривых в 3D",
        labels=("Оригинал", "Реконструкция")
    )
    plt.show()
    
    # Вычисление метрик
    metrics = calculate_reconstruction_metrics(original_curve, reconstructed_curve)
    print("Метрики качества реконструкции:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.6f}")
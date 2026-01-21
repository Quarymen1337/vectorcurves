import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class MoleculeCurveAnalyzer:
    """
    Класс для анализа кривой, описывающей молекулу.
    Вычисляет статистические характеристики и градиенты.
    """
    
    def __init__(self, curve_data: np.ndarray):
        """
        Инициализация класса с данными кривой.
        
        Parameters:
        -----------
        curve_data : np.ndarray
            Массив из точек, описывающих молекулу
        """
        self.curve_data = np.array(curve_data, dtype=np.float64)
        self._gradient: Optional[np.ndarray] = None
        self._characteristics: Optional[Dict] = None
    
    def _calculate_gradient(self) -> np.ndarray:
        """
        Вычисление градиента (производной) кривой.
        """
        if self._gradient is None:
            # Используем центральные разности для внутренних точек
            gradient = np.zeros_like(self.curve_data)
            
            # Центральные разности для внутренних точек
            gradient[1:-1] = (self.curve_data[2:] - self.curve_data[:-2]) / 2
            
            # Вперед/назад для граничных точек
            gradient[0] = self.curve_data[1] - self.curve_data[0]
            gradient[-1] = self.curve_data[-1] - self.curve_data[-2]
            
            self._gradient = gradient
        
        return self._gradient
    
    def calculate_characteristics(self) -> Dict[str, float]:
        """
        Вычисление всех характеристик кривой.
        
        Returns:
        --------
        Dict[str, float]: Словарь с характеристиками
        """
        if self._characteristics is not None:
            return self._characteristics
        
        data = self.curve_data
        gradient = self._calculate_gradient()
        
        # Основные статистики
        characteristics = {
            'len': len(data),
            'mean': np.mean(data),
            'median': np.median(data),
            'std': np.std(data, ddof=1),  # несмещенная оценка
            'variance': np.var(data, ddof=1),
            'min': np.min(data),
            'max': np.max(data),
            'range': np.max(data) - np.min(data)
        }
        
        # Квантили
        characteristics['q25'] = np.percentile(data, 25)
        characteristics['q75'] = np.percentile(data, 75)
        characteristics['iqr'] = characteristics['q75'] - characteristics['q25']
        
        # Асимметрия (скошенность)
        # Используем формулу Фишера-Пирсона
        mean = characteristics['mean']
        std = characteristics['std']
        n = len(data)
        if std > 0:
            characteristics['skewness'] = (np.sum((data - mean) ** 3) / n) / (std ** 3)
        else:
            characteristics['skewness'] = 0.0
        
        # Характеристики градиента
        characteristics['mean_gradient'] = np.mean(gradient)
        characteristics['max_gradient'] = np.max(gradient)
        characteristics['min_gradient'] = np.min(gradient)
        characteristics['std_gradient'] = np.std(gradient, ddof=1)
        
        # Полная вариация (Total Variation)
        characteristics['total_variation'] = np.sum(np.abs(gradient))
        
        self._characteristics = characteristics
        return characteristics
    
    def get_characteristic(self, name: str) -> float:
        """
        Получение конкретной характеристики по имени.
        
        Parameters:
        -----------
        name : str
            Имя характеристики
        
        Returns:
        --------
        float: Значение характеристики
        """
        if self._characteristics is None:
            self.calculate_characteristics()
        
        if name not in self._characteristics:
            raise KeyError(f"Характеристика '{name}' не найдена. "
                          f"Доступные характеристики: {list(self._characteristics.keys())}")
        
        return self._characteristics[name]
    
    def print_summary(self, precision: int = 6) -> None:
        """
        Вывод сводки характеристик в читаемом формате.
        
        Parameters:
        -----------
        precision : int
            Количество знаков после запятой
        """
        if self._characteristics is None:
            self.calculate_characteristics()
        
        print("=" * 60)
        print("ХАРАКТЕРИСТИКИ КРИВОЙ МОЛЕКУЛЫ")
        print("=" * 60)
        
        # Форматирование значений
        for key, value in self._characteristics.items():
            if isinstance(value, float):
                print(f"{key:20}: {value:.{precision}f}")
            else:
                print(f"{key:20}: {value}")
        
        print("=" * 60)
    
    def get_characteristics_table(self) -> List[Tuple[str, float]]:
        """
        Возвращает характеристики в виде таблицы (списка кортежей).
        
        Returns:
        --------
        List[Tuple[str, float]]: Список (название, значение)
        """
        if self._characteristics is None:
            self.calculate_characteristics()
        
        return [(key, value) for key, value in self._characteristics.items()]
    
    def get_gradient_data(self) -> np.ndarray:
        """
        Возвращает вычисленный градиент.
        
        Returns:
        --------
        np.ndarray: Массив градиента
        """
        return self._calculate_gradient()
    
    @property
    def gradient(self) -> np.ndarray:
        """Свойство для доступа к градиенту."""
        return self._calculate_gradient()
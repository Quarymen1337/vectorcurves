from src.preprocessing.curves import get_invariant_features
from src.preprocessing.interpolation import resample_trajectory_cubic_spline
from typing import List, Tuple, Dict, Any
import numpy as np
from tqdm import tqdm 
from src.preprocessing.curves import get_invariant_features
from src.preprocessing.interpolation import resample_trajectory_cubic_spline
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from tqdm import tqdm 

def resample_trajectories(ca_points: List[np.ndarray], 
                          k_points: int = 512,
                          interpolate: bool = True) -> List[np.ndarray]:
    """
    Ресемплирует траектории CA-атомов с использованием кубических сплайнов.
    
    Args:
        ca_points: Список массивов координат CA-атомов
        k_points: Количество точек после ресемплинга
        interpolate: Если True - интерполирует траекторию до k_points точек,
                    Если False - возвыыращает оригинальные точки в новых координатах
                    
    Returns:
        List[np.ndarray]: Список ресемплированных траекторий
    """
    resampled_coords = []
    
    if interpolate:
        print(f"Ресемплирование траекторий до {k_points} точек...")
    else:
        print(f"Преобразование траекторий в инвариантные признаки (без интерполяции)...")
    
    for i, example in enumerate(tqdm(ca_points, desc="Обработка")):
        try:
            # Получаем инвариантные признаки
            features = get_invariant_features(np.array(example))
            
            if interpolate:
                # Ресемплируем с помощью кубических сплайнов
                resampled = resample_trajectory_cubic_spline(features, k=k_points)
            else:
                # Возвращаем оригинальные точки в новых координатах
                # (просто преобразованные в инвариантные признаки)
                resampled = features
            
            resampled_coords.append(resampled)
            
        except Exception as e:
            print(f"Ошибка при обработке примера {i}: {e}")
            # В случае ошибки можно добавить None или пропустить
            resampled_coords.append(None)
    
    return resampled_coords
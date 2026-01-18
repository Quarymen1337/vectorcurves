from src.utils.config_loader import project_config
from typing import List, Tuple, Dict, Any

def load_config() -> Dict[str, Any]:
    """
    Загружает конфигурацию проекта из YAML файла.
    
    Returns:
        Dict[str, Any]: Словарь с конфигурацией проекта
    """
    config = project_config.load()
    
    print("\n" + "="*60)
    print("КОНФИГУРАЦИЯ ПРОЕКТА")
    print("="*60)
    print(f"   Проект: {config.get('project', {}).get('name', 'N/A')}")
    print(f"   Версия: {config.get('project', {}).get('version', 'N/A')}")
    print(f"   Автор: {config.get('project', {}).get('author', 'N/A')}")
    print(f"   Датасет: {config.get('dataset', {}).get('path', 'N/A')}")
    print(f"   Устройство: {config.get('model', {}).get('device', 'cpu')}")
    print("="*60 + "\n")
    
    return config

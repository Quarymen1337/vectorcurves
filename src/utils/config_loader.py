"""
Универсальный загрузчик конфигов для проекта
"""
import yaml
import os
from pathlib import Path
from typing import Any, Dict, Optional
from functools import lru_cache

class ProjectConfig:
    """Загрузчик конфигов для проекта"""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or self._find_project_root()
        self.configs_dir = self.project_root / "configs"
        
        # Кэш конфигов
        self._config_cache = {}
    
    def _find_project_root(self) -> Path:
        """Находит корень проекта"""
        current = Path.cwd()
        
        # Если в notebooks
        if current.name == "notebooks":
            return current.parent
        
        # Ищем папку с src и configs
        for parent in [current] + list(current.parents):
            if (parent / "src").exists() and (parent / "configs").exists():
                return parent
        
        return current
    
    @lru_cache(maxsize=32)
    def load(self, config_path: str = "config.yaml") -> Dict[str, Any]:
        """Загружает конфиг с кэшированием"""
        full_path = self.configs_dir / config_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"Config not found: {full_path}")
        
        with open(full_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Обрабатываем includes
        if 'include' in config:
            includes = config.pop('include')
            if isinstance(includes, str):
                includes = [includes]
            
            merged = {}
            for include in includes:
                included_config = self.load(include)
                merged.update(included_config)
            
            merged.update(config)
            config = merged
        
        # Заменяем переменные окружения
        config = self._replace_env_vars(config)
        
        return config
    
    def _replace_env_vars(self, obj: Any) -> Any:
        """Рекурсивно заменяет ${VAR} на значения из окружения"""
        if isinstance(obj, dict):
            return {k: self._replace_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._replace_env_vars(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            var_name = obj[2:-1]
            return os.getenv(var_name, obj)
        else:
            return obj
    
    def get(self, config_path: str = "config.yaml", key: str = None, default: Any = None) -> Any:
        """Получает значение из конфига"""
        config = self.load(config_path)
        
        if key is None:
            return config
        
        # Поддержка dot notation
        keys = key.split('.')
        value = config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value

# Создаем глобальный инстанс
project_config = ProjectConfig()
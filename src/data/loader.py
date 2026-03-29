"""
Загрузчик PDB файлов и предобработка данных
"""

import os
from typing import List, Tuple
from tqdm import tqdm
from src.pdb.PDBFile import PDBFile


def load_and_preprocess_pdb_files(
    dataset_path: str,
    min_atoms: int = 0,
    max_atoms: int = 1000000,
    enable_warnings: bool = False
) -> Tuple[List[str], List[List[List[float]]]]:
    """
    Загружает и предобрабатывает PDB файлы из директории.

    Args:
        dataset_path: Путь к директории с PDB файлами
        min_atoms: Минимальное количество атомов
        max_atoms: Максимальное количество атомов
        enable_warnings: Включить предупреждения

    Returns:
        Tuple[List[str], List[List[List[float]]]]:
            - Список путей к файлам
            - Список координат CA-атомов
    """
    pdb_files = []
    ca_points = []

    for root, dirs, files in os.walk(dataset_path):
        for file in tqdm(files, desc="Загрузка PDB файлов"):
            if not file.lower().endswith('.pdb'):
                continue

            file_path = os.path.join(root, file)

            try:
                pdb_file = PDBFile(file_path, enable_warnings=enable_warnings)

                if len(pdb_file.atom_line) > 0:
                    n_atoms = len(pdb_file.ca_atom_line)
                    if min_atoms <= n_atoms <= max_atoms:
                        backbone_coords, _ = pdb_file.get_backbone_cloud()
                        pdb_files.append(file_path)
                        ca_points.append(backbone_coords)
            except Exception as e:
                if enable_warnings:
                    print(f"Ошибка при обработке {file_path}: {e}")

    return pdb_files, ca_points

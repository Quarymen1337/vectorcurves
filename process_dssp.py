#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Скрипт для обработки DSSP (Dictionary of Secondary Structure in Proteins) данных
на основе notebook marking_data.ipynb
"""

import os
import subprocess
import pandas as pd
import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def run_dssp(pdb_file, output_dssp_file):
    """
    Запуск DSSP с различными вариантами синтаксиса
    
    Args:
        pdb_file (str): Путь к PDB файлу
        output_dssp_file (str): Путь к выходному DSSP файлу
    
    Returns:
        bool: True если DSSP успешно запущен, иначе False
    """
    # Вариант 1: Без флагов (просто входной и выходной файлы)
    try:
        cmd = ['dssp', pdb_file, output_dssp_file]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("DSSP запущен успешно (вариант 1)")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Вариант 1 не сработал: {e}")
    
    # Вариант 2: С флагом -i для ввода (если поддерживается)
    try:
        cmd = ['dssp', '-i', pdb_file, output_dssp_file]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("DSSP запущен успешно (вариант 2)")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Вариант 2 не сработал или команда не найдена")
    
    # Вариант 3: Попробуем mkdssp (новое имя исполняемого файла DSSP)
    try:
        cmd = ['mkdssp', pdb_file, output_dssp_file]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("mkdssp запущен успешно")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"mkdssp не сработал: {e}")
    
    # Вариант 4: Попробуем с перенаправлением вывода в файл
    try:
        with open(output_dssp_file, 'w') as f:
            cmd = ['dssp', pdb_file]
            result = subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.PIPE, text=True)
        print("DSSP запущен успешно (вариант 4)")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Вариант 4 не сработал: {e}")
    
    return False

def extract_dssp_features(pdb_file_path):
    """
    Извлечение DSSP признаков из PDB файла
    
    Args:
        pdb_file_path (str): Путь к PDB файлу
    
    Returns:
        DataFrame or None: DataFrame с DSSP признаками или None в случае ошибки
    """
    # Создаем временный файл для DSSP
    temp_dssp_file = "temp_processing.dssp"
    
    # Запускаем DSSP
    if not run_dssp(pdb_file_path, temp_dssp_file):
        print(f"Не удалось запустить DSSP для файла {pdb_file_path}")
        return None
    
    try:
        # Парсим PDB файл
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_file_path)
        model = structure[0]
        
        # Получаем DSSP данные
        dssp = DSSP(model, temp_dssp_file, file_type='DSSP')
        
        # Извлекаем данные
        dssp_data = []
        for key in dssp.keys():
            dssp_data.append({
                'chain': key[0],
                'residue_num': key[1][1],
                'residue_name': dssp[key][1],
                'secondary_structure': dssp[key][2],
                'ASA': dssp[key][3],  # Accessible Surface Area
                'phi': dssp[key][4],
                'psi': dssp[key][5],
                'tau': dssp[key][6],
                'theta': dssp[key][7]
            })
        
        # Удаляем временный файл
        if os.path.exists(temp_dssp_file):
            os.remove(temp_dssp_file)
        
        # Создаем DataFrame
        df = pd.DataFrame(dssp_data)
        return df
        
    except Exception as e:
        print(f"Ошибка при обработке DSSP данных для {pdb_file_path}: {e}")
        # Удаляем временный файл в случае ошибки
        if os.path.exists(temp_dssp_file):
            os.remove(temp_dssp_file)
        return None

def process_pdb_directory(pdb_directory, output_csv=None):
    """
    Обработка всех PDB файлов в директории с извлечением DSSP признаков
    
    Args:
        pdb_directory (str): Директория с PDB файлами
        output_csv (str, optional): Путь к CSV файлу для сохранения результатов
    
    Returns:
        list: Список DataFrames с DSSP признаками для каждого файла
    """
    # Найдем все PDB файлы
    pdb_files = []
    for root, dirs, files in os.walk(pdb_directory):
        for file in files:
            if file.lower().endswith('.pdb'):
                pdb_files.append(os.path.join(root, file))
    
    print(f"Найдено {len(pdb_files)} PDB файлов для обработки")
    
    results = []
    processed_files = []
    
    for pdb_file in tqdm(pdb_files, desc="Обработка PDB файлов"):
        try:
            dssp_df = extract_dssp_features(pdb_file)
            if dssp_df is not None and not dssp_df.empty:
                dssp_df['source_file'] = pdb_file  # Добавим путь к исходному файлу
                results.append(dssp_df)
                processed_files.append(pdb_file)
            else:
                print(f"Не удалось извлечь DSSP данные для {pdb_file}")
        except Exception as e:
            print(f"Ошибка при обработке файла {pdb_file}: {e}")
    
    print(f"Успешно обработано {len(results)} файлов из {len(pdb_files)}")
    
    # Сохраняем результаты в CSV, если указан путь
    if output_csv and results:
        combined_df = pd.concat(results, ignore_index=True)
        combined_df.to_csv(output_csv, index=False)
        print(f"Результаты сохранены в {output_csv}")
    
    return results, processed_files

def main():
    """
    Основная функция для демонстрации работы скрипта
    """
    # Пример использования
    pdb_directory = "data/new_dompdb/dompdb"  # Адаптируйте под вашу структуру
    
    # Проверим, существует ли директория
    if not os.path.exists(pdb_directory):
        print(f"Директория {pdb_directory} не найдена. Пожалуйста, укажите правильный путь.")
        # Попробуем найти PDB файлы в текущей директории
        current_dir = os.getcwd()
        print(f"Проверяем текущую директорию: {current_dir}")
        pdb_directory = current_dir
    
    # Обрабатываем директорию
    results, processed_files = process_pdb_directory(
        pdb_directory, 
        output_csv="dssp_features.csv"
    )
    
    # Выводим информацию о первом успешном результате
    if results:
        print("\nПример DSSP признаков из первого файла:")
        print(results[0].head())
        print(f"\nКоличество остатков в первом файле: {len(results[0])}")
        print(f"Типы вторичной структуры: {results[0]['secondary_structure'].unique()}")
    
    return results, processed_files

if __name__ == "__main__":
    results, files = main()
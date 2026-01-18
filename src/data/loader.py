import json
import numpy as np
import pandas as pd
import torch
import os
from tqdm import tqdm
from src.preprocessing.interpolation import resample_trajectory, resample_trajectory_cubic_spline
from src.preprocessing.atom_process import get_ca_coordinates
from src.preprocessing.curves import get_invariant_features
import torch_geometric
import torch
from torch_geometric.data import Data
import torch
from src.pdb.PDBFile import PDBFile
from typing import List, Tuple, Dict, Any



def process_protonated_files(directory, max_l = 2000, alpha = 2000, beta = 5000):
    len_dataset = 0
    skiped_molecule = 0
    print('Начало подготовки. Загрузка файлов')
    results = []
    
    protonated_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.protonated.json'):
                full_path = os.path.join(root, file)
                protonated_files.append(full_path)
    
    print(f"Найдено {len(protonated_files)} protonated файлов")
    
    for file_path in tqdm(protonated_files, desc="Обработка файлов"):
        directory_path = os.path.dirname(file_path)
        filename = os.path.basename(file_path)
        
        chain_name = filename.replace('_chain.protonated.json', '')
        
        sasa_file = os.path.join(directory_path, f"{chain_name}_sasa.json")
        charges_file = os.path.join(directory_path, f"{chain_name}_charges.json")
        
        if all(os.path.exists(f) for f in [file_path, sasa_file, charges_file]):
            try:
                with open(file_path, 'r') as f:
                    protonated_data = json.load(f)
                with open(sasa_file, 'r') as f:
                    sasa_data = json.load(f)
                with open(charges_file, 'r') as f:
                    charges_data = json.load(f)
                
                if (len(protonated_data['atoms']) < alpha) or (len(protonated_data['atoms']) > beta):
                    skiped_molecule += 1
                    continue
                    
                results.append({
                    'chain': protonated_data,
                    'sasa': sasa_data,
                    'charges': charges_data,
                    'file': file_path,
                    'sasa_file': sasa_file,
                    'charges_file': charges_file,
                    'chain_name': chain_name
                })
                len_dataset += 1
                if len_dataset > max_l:
                    break
                
            except json.JSONDecodeError as e:
                print(f"Ошибка чтения JSON в файле {file_path}: {e}")
        else:
            print(f"Не все файлы существуют для {file_path}")
            missing_files = []
            if not os.path.exists(file_path):
                missing_files.append("protonated")
            if not os.path.exists(sasa_file):
                missing_files.append("sasa")
            if not os.path.exists(charges_file):
                missing_files.append("charges")
            print(f"Отсутствуют файлы: {missing_files}")
    
    print(f"Успешно обработано {len(results)} файлов")
    print(f"Не прошли критериев {skiped_molecule} молекул")

    return results


def create_data(JSON_DATA, K = 256):
    trace_data = []
    graph_data = []
    ca_coordinate_data = []

    print('Начало расчета координат')
    for i in tqdm(range(len(JSON_DATA))):
        ca_coord = get_ca_coordinates(JSON_DATA[i])
        if len(ca_coord) < 1:
            continue
        ca_coordinate_data.append(ca_coord)
        
        resampled_coords = resample_trajectory_cubic_spline(get_invariant_features(np.array(ca_coord)), k = K)
        trace_data.append(resampled_coords)
        atoms_data = JSON_DATA[i]['chain']['atoms']
        sasa = JSON_DATA[i]['sasa']['sasa']
        charges = JSON_DATA[i]['charges']['charges']
        bonds = JSON_DATA[i]['chain']['bonds']
        
        x = torch.tensor([
            [atom['x'], atom['y'], atom['z'], atom['radii'], atom['temp_factor'], sasa[value], charges[value]] 
            for value, atom in enumerate(atoms_data)
        ])

        edge_index = torch.tensor([row[:2] for row in bonds])
        edge_attr = torch.tensor([row[2] for row in bonds])
        graph = Data(x=x, edge_index=edge_index.T, edge_attr = edge_attr.unsqueeze(0))
        graph_data.append(graph)
    print('Successful!')
    return trace_data, graph_data, ca_coordinate_data

def load_and_preprocess_pdb_files(dataset_path: str) -> Tuple[List[PDBFile], List[np.ndarray]]:
    """
    Загружает PDB файлы и извлекает CA-атомы.
    
    Args:
        dataset_path: Путь к директории с PDB файлами
        
    Returns:
        Tuple[List[PDBFile], List[np.ndarray]]: Списки PDB объектов и координат CA-атомов
    """
    pdb_files = []
    ca_points = []
    zero_error_count = 0
    
    print("Загрузка PDB файлов...")
    
    for root, dirs, files in os.walk(dataset_path):
        for file in tqdm(files, desc="Обработка файлов"):
            file_path = os.path.join(root, file)
            
            try:
                pdb_file = PDBFile(file_path)
                
                if len(pdb_file.atom_line) > 0:
                    ca_point = pdb_file.get_point_cloud()
                    pdb_files.append(pdb_file)
                    ca_points.append(ca_point)
                else:
                    zero_error_count += 1
                    
            except Exception as e:
                print(f"Ошибка при обработке файла {file}: {e}")
                zero_error_count += 1
    
    print(f"\nЗагружено {len(pdb_files)} файлов, ошибок: {zero_error_count}")
    
    return pdb_files, ca_points


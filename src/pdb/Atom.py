import os
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from pathlib import Path

@dataclass
class Atom:
    record_type: str      
    serial: int           
    name: str            
    alt_loc: str          
    res_name: str         
    chain_id: str         
    res_seq: int          
    x: float              
    y: float              
    z: float              
    occupancy: float      
    b_factor: float      
    
    @classmethod
    def from_pdb_line(cls, line: str):
        if not line.strip():
            return None
            
        record_type = line[0:6].strip()
        if record_type not in ['ATOM', 'HETATM']:
            return None
        
        try:
            serial = int(line[6:11].strip())
            name = line[12:16].strip()
            alt_loc = line[16:17].strip()
            res_name = line[17:20].strip()
            chain_id = line[21:22].strip()
            res_seq = int(line[22:26].strip())
        
            # ВАЖНО: используем фиксированные позиции из стандарта PDB
            # X занимает символы 30-37 (включительно)
            x_str = line[30:38]  # 30..37
            y_str = line[38:46]  # 38..45  
            z_str = line[46:54]  # 46..53
            
            x = float(x_str.strip())
            y = float(y_str.strip())
            z = float(z_str.strip())
            
            # Остальные поля тоже по фиксированным позициям
            occupancy_str = line[54:60]
            b_factor_str = line[60:66]
            
            occupancy = float(occupancy_str.strip() or 1.0)
            b_factor = float(b_factor_str.strip() or 0.0)
            
            return cls(
                record_type=record_type,
                serial=serial,
                name=name,
                alt_loc=alt_loc,
                res_name=res_name,
                chain_id=chain_id,
                res_seq=res_seq,
                x=x,
                y=y,
                z=z,
                occupancy=occupancy,
                b_factor=b_factor
            )
            
        except (ValueError, IndexError) as e:
            print(f"Ошибка парсинга строки: {line.strip()}")
            print(f"Позиции: x={x_str!r}, y={y_str!r}, z={z_str!r}")
            print(f"Ошибка: {e}")
            return None
    
    def to_pdb_line(self) -> str:
        return (f"{self.record_type:6}{self.serial:5} {self.name:^4}{self.alt_loc:1}"
                f"{self.res_name:3} {self.chain_id:1}{self.res_seq:4}   "
                f"{self.x:8.3f}{self.y:8.3f}{self.z:8.3f}"
                f"{self.occupancy:6.2f}{self.b_factor:6.2f}")
    
    def get_coordinates(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])
    
    def get_residue_id(self) -> str:
        return f"{self.chain_id}:{self.res_name}{self.res_seq}"
    
    def __str__(self) -> str:
        return (f"{self.record_type} {self.serial} {self.name} "
                f"{self.res_name} {self.chain_id}{self.res_seq} "
                f"({self.x:.3f}, {self.y:.3f}, {self.z:.3f})")

import warnings
from src.pdb.Atom import Atom

class PDBFile:
    def __init__(self, PATH, enable_warnings=True):
        """
        Инициализация объекта PDBFile.
        
        Args:
            PATH (str): Путь к PDB файлу
            enable_warnings (bool): Флаг включения/отключения предупреждений
        """
        self.path = PATH
        self.atom_line = []  # Список всех атомов
        self.ca_atom_line = []  # Список только CA атомов (альфа-углероды)
        self.backbone_line = []
        self.backbone_type_line = []
        self.enable_warnings = enable_warnings
        
        # Чтение файла и поиск CA атомов при инициализации
        self.read(self.path)
        self.find_ca()
    def read(self, PATH):
        """
        Чтение PDB файла и извлечение атомов.
        
        Args:
            PATH (str): Путь к PDB файлу
        """
        with open(PATH, 'r') as file:
            for line in file:
                # Обрабатываем строки, содержащие информацию об атомах
                if ('ATOM' in line) or ('HETATM' in line):
                    atom = Atom.from_pdb_line(line)
                    self.atom_line.append(atom)

    def find_ca(self):
        """
        Поиск CA атомов (альфа-углеродов) в списке всех атомов.
        Используется для анализа белковых структур.
        """
        if len(self.atom_line) == 0:
            self._show_warning('ERROR! Length atom_line = 0!!!', self.path)
        else:
            # Фильтрация CA атомов
            for atom in self.atom_line:
                if atom.name == 'CA':
                    self.ca_atom_line.append(atom)
            
            # Проверка наличия CA атомов
            if len(self.ca_atom_line) == 0:
                self._show_warning('Not Found CA atoms')

    def find_backbone(self):
        if len(self.atom_line) == 0:
            self._show_warning('ERROR! Length atom_line = 0!!!', self.path)
        else:
            for atom in self.atom_line:
                if atom.name == 'CA':
                    self.backbone_line.append(atom)
                    self.backbone_type_line.append(0)
                elif atom.name == 'N':
                    self.backbone_line.append(atom)
                    self.backbone_type_line.append(1)
                elif atom.name == 'C':
                    self.backbone_line.append(atom)
                    self.backbone_type_line.append(2)
                
                
            
            if len(self.backbone_line) == 0:
                self._show_warning('Not Found backbone atoms')

        return self.backbone_line, self.backbone_type_line
                


    def _show_warning(self, message, path=None):
        """
        Вспомогательный метод для показа предупреждений.
        
        Args:
            message (str): Текст предупреждения
            path (str, optional): Путь к файлу для дополнительной информации
        """
        if self.enable_warnings:
            full_message = message
            if path:
                full_message = f"{message}\nPath: {path}"
            warnings.warn(full_message, category=UserWarning, stacklevel=3)

    def get_point_cloud(self, only_ca=True):
        """
        Преобразование координат атомов в облако точек.
        
        Args:
            only_ca (bool): Если True, возвращаются только CA атомы,
                           иначе возвращаются все атомы
                           
        Returns:
            list: Список координат точек [[x1, y1, z1], [x2, y2, z2], ...]
        """
        points = []
        atom_array = None
        
        # Выбор массива атомов в зависимости от параметра
        if only_ca: 
            atom_array = self.ca_atom_line
        else:
            atom_array = self.atom_line

        # Извлечение координат атомов
        for atom in atom_array:
            x = atom.x
            y = atom.y
            z = atom.z
            points.append([x, y, z])
        return points

    def get_backbone_cloud(self):
        #Возвращение бекбона с типами атомов
        self.find_backbone()
        points = []
        atom_array = None
        
    
        atom_array = self.backbone_line
    

        # Извлечение координат атомов
        for atom in atom_array:
            x = atom.x
            y = atom.y
            z = atom.z
            points.append([x, y, z])
        return points, self.backbone_type_line

    
    
    def getAtom(self):
        """
        Получение списка всех атомов.
        
        Returns:
            list: Список объектов Atom
        """
        return self.atom_line
    
    def getCaAtom(self):
        """
        Получение списка CA атомов.
        
        Returns:
            list: Список объектов Atom, представляющих CA атомы
        """
        return self.ca_atom_line

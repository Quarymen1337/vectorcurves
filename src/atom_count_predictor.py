from src.features.CurveAnalyser import MoleculeCurveAnalyzer
from src.data.loader import load_and_preprocess_pdb_files
from src.preprocessing.resample import resample_trajectories
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import math

class AtomCountPredictor():
    def __init__(self, data_path:str, k_intp:int = 10000, test_size:float = 0.2, retrain_model = False):
        self.k_points = k_intp
        print('Загрузка')
        self.ca_points = self._load_data_(data_path)


        print('Ресемплинг')
        intp_resampled_coords, no_intp_resampled_coords = self._resample_(self.ca_points,  self.k_points)
        
        self.intp = intp_resampled_coords
        self.no_intp = no_intp_resampled_coords        

        self.df = self._generate_feature_()
        self.model = None
        if retrain_model:
            print('Обучение модели')
            self.model = self.fit_linear_regression(test_size,test_size )

    def _load_data_(self,dataset_path):
        pdb_files, ca_points = load_and_preprocess_pdb_files(dataset_path)
        return ca_points


    def _curve_length_3d_(self,points):
        if len(points) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(len(points) - 1):
            x1, y1, z1 = points[i]
            x2, y2, z2 = points[i + 1]
            
            dx = x2 - x1
            dy = y2 - y1
            dz = z2 - z1
            
            segment_length = math.sqrt(dx*dx + dy*dy + dz*dz)
            total_length += segment_length
        
        return total_length
        
    def _resample_(self, ca_points, k_points):
        no_intp_resampled_coords = resample_trajectories(ca_points, k_points, interpolate = False) #Оригинальные точки в новых координатах
        intp_resampled_coords =  resample_trajectories(ca_points, k_points, interpolate = True) #Интерполированныее точки в новых координатах
        return intp_resampled_coords, no_intp_resampled_coords

    def _generate_feature_(self):
        len_curves_array = []
        for sample in tqdm(self.intp):
            len_curves_array.append(self._curve_length_3d_(sample))
            
        atom_counts = []
        for sample in tqdm(self.no_intp):
            atom_counts.append(len(sample))

        array_of_stats = []
        for sample in tqdm(self.intp):
            array_of_stats.append(MoleculeCurveAnalyzer(sample).get_characteristics_table())

        data_list = []
        for stats in array_of_stats:
            data_dict = {metric: value for metric, value in stats}
            data_list.append(data_dict)
        
        df = pd.DataFrame(data_list)

        df['len'] = len_curves_array
        df['target'] = atom_counts

        return df
    def get_dataset(self):
        return self.df

    def fit_linear_regression(self, test_size, random_state):
        df = self.df
        
        X = df.drop(['target'], axis=1)
        y = df['target']               
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2,
            random_state=42
        )
        
        model = LinearRegression()
        model.fit(X_train, y_train)

        
        y_test_pred = model.predict(X_test)
        print(f"Оценка на валидации MAE: {mean_absolute_error(y_test, y_test_pred):.4f}")
        print("Переобучение на полном наборе данных")

        model = LinearRegression()
        model.fit(X, y)

        self.model = model

        return model

        
    def load_model(self, pickle_model_path):
        with open(pickle_model_path, 'rb') as f:
            loaded_model = pickle.load(f)
            self.model = loaded_model
            
    def predict_atom_count(self,sample_X):
        if self.model == None:
            print('Не найдена модель, необходимо загрузить(load model) или переобучить(fit_linear_regression)')
        else:
            predict = self.model(sample_X)
        return predict
        
        


            

        


        
        
    
    
        
        

    
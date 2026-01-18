import pandas as pd
from preprocessing.curves import get_invariant_features


#Corr-matrix
def coordinate_coor_matrix(original_points):
    df = pd.DataFrame(original_points, columns=['Feature1', 'Feature2', 'Feature3'])

    correlation_matrix = df.corr()
    print('Corr матрица для изначальных точек')
    print(correlation_matrix)
    invariant_coord = get_invariant_features(original_points)

    invariant_df = pd.DataFrame(invariant_coord, columns=['Feature1', 'Feature2', 'Feature3'])
    invariant_correlation_matrix = invariant_df.corr()

    print('Corr матрица для инвариантных точек')
    print(invariant_correlation_matrix)
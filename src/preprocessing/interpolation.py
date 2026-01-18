import numpy as np
from src.preprocessing.splines.cubic_spline import CubicSpline3D
# Равномерная интерполяция
def resample_trajectory(original_points, k):
    n_original_points = original_points.shape[0]
    n_dimensions = original_points.shape[1]
    
    original_time = np.linspace(0, 1, n_original_points)
    new_time = np.linspace(0, 1, k)
    resampled_points = np.zeros((k, n_dimensions))
    for i in range(n_dimensions):
        resampled_points[:, i] = np.interp(new_time, original_time, original_points[:, i])
    return resampled_points

# Интерполяция сплайнами
def resample_trajectory_cubic_spline(original_points, k):
    spline3d = CubicSpline3D(np.array(original_points))
    N = k
    curve_pts = spline3d.sample(N)
    return np.array(curve_pts)


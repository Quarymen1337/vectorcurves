import numpy as np

# Generate sample coordinate data for testing

# 1. Normal coordinates (good data)
coords_normal = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.5, 0.1],
    [2.0, 0.2, 0.3],
    [3.0, 0.8, 0.2],
    [4.0, 0.1, 0.4],
    [5.0, 0.6, 0.1],
    [6.0, 0.3, 0.5],
    [7.0, 0.7, 0.2],
    [8.0, 0.4, 0.6],
    [9.0, 0.9, 0.3]
])

np.save('/home/main/repos/b_splines/tests/data/coordinates/normal_coords.npy', coords_normal)

# 2. Coordinates with NaN values
coords_with_nan = coords_normal.copy()
coords_with_nan[2, 1] = np.nan  # Add a NaN value
coords_with_nan[5, 0] = np.nan  # Add another NaN value

np.save('/home/main/repos/b_splines/tests/data/coordinates/coords_with_nan.npy', coords_with_nan)

# 3. Coordinates with infinite values
coords_with_inf = coords_normal.copy()
coords_with_inf[3, 2] = np.inf  # Add an infinite value

np.save('/home/main/repos/b_splines/tests/data/coordinates/coords_with_inf.npy', coords_with_inf)

# 4. Coordinates with outliers
coords_with_outliers = coords_normal.copy()
coords_with_outliers[1, 0] = 500.0  # Add an outlier
coords_with_outliers[7, 1] = -300.0  # Add another outlier

np.save('/home/main/repos/b_splines/tests/data/coordinates/coords_with_outliers.npy', coords_with_outliers)

# 5. Coordinates with chain breaks (irregular numbering)
coords_chain_breaks = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.5, 0.1],
    [2.0, 0.2, 0.3],
    # Break in sequence (skip residue 3)
    [4.0, 0.8, 0.2],  # This would be residue 4 after residue 2
    [5.0, 0.1, 0.4],
    [6.0, 0.6, 0.1],
    # Another break
    [8.0, 0.3, 0.5],  # This would be residue 8 after residue 6
    [9.0, 0.7, 0.2]
])

np.save('/home/main/repos/b_splines/tests/data/coordinates/coords_chain_breaks.npy', coords_chain_breaks)

# 6. Coordinates with short CA-CA distances
coords_short_distances = np.array([
    [0.0, 0.0, 0.0],
    [0.5, 0.0, 0.0],  # Only 0.5A from previous, very short
    [0.7, 0.2, 0.1],  # Short distance
    [1.0, 0.5, 0.3], 
    [1.3, 0.8, 0.5], 
    [1.6, 1.1, 0.7]  # Normal distances
])

np.save('/home/main/repos/b_splines/tests/data/coordinates/coords_short_distances.npy', coords_short_distances)

# 7. Coordinates with long CA-CA distances
coords_long_distances = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0], 
    [8.0, 0.5, 0.2],  # 7A from previous, very long
    [8.3, 0.8, 0.4], 
    [8.6, 1.1, 0.6], 
    [9.0, 1.4, 0.8]   # Then normal distances
])

np.save('/home/main/repos/b_splines/tests/data/coordinates/coords_long_distances.npy', coords_long_distances)

print("Sample coordinate files created successfully!")
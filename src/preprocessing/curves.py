import numpy as np

def get_invariant_features(points):
    points = np.array(points)
    
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    
    cov_matrix = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    for i in range(3):
        if eigenvectors[i, 0] < 0:
            eigenvectors[:, i] *= -1
    
    aligned_points = centered @ eigenvectors
    
    return aligned_points
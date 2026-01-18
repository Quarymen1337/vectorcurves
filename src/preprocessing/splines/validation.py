"""
Validation module for coordinates and protein structure data.
Provides functions to validate coordinate data, check for gaps,
validate CA-CA distances, and generate quality reports.
"""

import numpy as np
from typing import List, Tuple, Dict, Any


def detect_nan_coordinates(coordinates: np.ndarray) -> List[int]:
    """
    Detect NaN values in coordinate array.
    
    Args:
        coordinates: Array of shape (n, 3) with x,y,z coordinates
        
    Returns:
        List of indices where NaN values are detected
    """
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("Coordinates must be a 2D array with shape (n, 3)")
    
    nan_indices = []
    for i, coord in enumerate(coordinates):
        if np.any(np.isnan(coord)):
            nan_indices.append(i)
    return nan_indices


def detect_inf_coordinates(coordinates: np.ndarray) -> List[int]:
    """
    Detect infinite values in coordinate array.
    
    Args:
        coordinates: Array of shape (n, 3) with x,y,z coordinates
        
    Returns:
        List of indices where infinite values are detected
    """
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("Coordinates must be a 2D array with shape (n, 3)")
    
    inf_indices = []
    for i, coord in enumerate(coordinates):
        if np.any(np.isinf(coord)):
            inf_indices.append(i)
    return inf_indices


def check_coordinate_ranges(coordinates: np.ndarray, min_val: float = -200.0, max_val: float = 200.0) -> List[int]:
    """
    Check if coordinates are within expected ranges (detect outliers).
    
    Args:
        coordinates: Array of shape (n, 3) with x,y,z coordinates
        min_val: Minimum allowed coordinate value
        max_val: Maximum allowed coordinate value
        
    Returns:
        List of indices where coordinates are outside the range
    """
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("Coordinates must be a 2D array with shape (n, 3)")
    
    outlier_indices = []
    for i, coord in enumerate(coordinates):
        if np.any(coord < min_val) or np.any(coord > max_val):
            outlier_indices.append(i)
    return outlier_indices


def filter_anomalous_coordinates(coordinates: np.ndarray, 
                                min_val: float = -200.0, 
                                max_val: float = 200.0) -> Tuple[np.ndarray, List[int]]:
    """
    Filter out anomalous coordinates.
    
    Args:
        coordinates: Array of shape (n, 3) with x,y,z coordinates
        min_val: Minimum allowed coordinate value
        max_val: Maximum allowed coordinate value
        
    Returns:
        Filtered coordinates and list of removed indices
    """
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("Coordinates must be a 2D array with shape (n, 3)")
    
    valid_mask = np.ones(len(coordinates), dtype=bool)
    for i, coord in enumerate(coordinates):
        if np.any(coord < min_val) or np.any(coord > max_val) or np.any(np.isnan(coord)) or np.any(np.isinf(coord)):
            valid_mask[i] = False
    
    removed_indices = [i for i, is_valid in enumerate(valid_mask) if not is_valid]
    filtered_coords = coordinates[valid_mask]
    
    return filtered_coords, removed_indices


def detect_chain_breaks(residue_numbers: List[int]) -> List[Tuple[int, int]]:
    """
    Detect breaks in residue numbering sequence.
    
    Args:
        residue_numbers: List of consecutive residue numbers
        
    Returns:
        List of (start_index, end_index) tuples indicating chain breaks
    """
    breaks = []
    for i in range(len(residue_numbers) - 1):
        if residue_numbers[i+1] - residue_numbers[i] != 1:
            breaks.append((i, i+1))
    return breaks


def create_residue_mapping(residue_numbers: List[int]) -> Dict[int, int]:
    """
    Create mapping from original residue numbers to continuous indices.
    
    Args:
        residue_numbers: List of residue numbers (may have gaps)
        
    Returns:
        Dictionary mapping original residue numbers to continuous indices
    """
    mapping = {}
    continuous_index = 0
    for res_num in residue_numbers:
        mapping[res_num] = continuous_index
        continuous_index += 1
    return mapping


def segment_chain(residue_numbers: List[int]) -> List[List[int]]:
    """
    Segment chain into continuous fragments.
    
    Args:
        residue_numbers: List of residue numbers (may have gaps)
        
    Returns:
        List of lists, each containing continuous residue number sequences
    """
    if not residue_numbers:
        return []
    
    segments = []
    current_segment = [residue_numbers[0]]
    
    for i in range(1, len(residue_numbers)):
        if residue_numbers[i] - residue_numbers[i-1] == 1:
            # Continuous sequence
            current_segment.append(residue_numbers[i])
        else:
            # Break in sequence
            segments.append(current_segment)
            current_segment = [residue_numbers[i]]
    
    # Add the last segment
    segments.append(current_segment)
    
    return segments


def calculate_ca_ca_distances(coordinates: np.ndarray) -> np.ndarray:
    """
    Calculate distances between consecutive CA atoms.
    
    Args:
        coordinates: Array of shape (n, 3) with CA coordinates
        
    Returns:
        Array of distances between consecutive CA atoms
    """
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("Coordinates must be a 2D array with shape (n, 3)")
    
    if len(coordinates) < 2:
        return np.array([])
    
    distances = []
    for i in range(len(coordinates) - 1):
        dist = np.linalg.norm(coordinates[i+1] - coordinates[i])
        distances.append(dist)
    
    return np.array(distances)


def detect_short_ca_ca_distances(distances: np.ndarray, threshold: float = 3.0) -> List[int]:
    """
    Detect unusually short CA-CA distances.
    
    Args:
        distances: Array of CA-CA distances
        threshold: Distance threshold below which is considered short
        
    Returns:
        List of indices where distances are too short
    """
    short_indices = []
    for i, dist in enumerate(distances):
        if dist < threshold:
            short_indices.append(i)
    return short_indices


def detect_long_ca_ca_distances(distances: np.ndarray, threshold: float = 4.5) -> List[int]:
    """
    Detect unusually long CA-CA distances.
    
    Args:
        distances: Array of CA-CA distances
        threshold: Distance threshold above which is considered long
        
    Returns:
        List of indices where distances are too long
    """
    long_indices = []
    for i, dist in enumerate(distances):
        if dist > threshold:
            long_indices.append(i)
    return long_indices


def identify_chain_breaks_by_distance(distances: np.ndarray, threshold: float = 4.5) -> List[int]:
    """
    Identify chain breaks based on CA-CA distances.
    
    Args:
        distances: Array of CA-CA distances
        threshold: Distance threshold above which indicates a chain break
        
    Returns:
        List of indices where chain breaks occur
    """
    break_indices = []
    for i, dist in enumerate(distances):
        if dist > threshold:
            break_indices.append(i)
    return break_indices


def generate_quality_report(coordinates: np.ndarray, 
                           residue_numbers: List[int] = None,
                           expected_ca_ca_distance: float = 3.8) -> Dict[str, Any]:
    """
    Generate a quality report with various metrics.
    
    Args:
        coordinates: Array of shape (n, 3) with coordinates
        residue_numbers: List of residue numbers (optional)
        expected_ca_ca_distance: Expected CA-CA distance for calculations
        
    Returns:
        Dictionary containing quality metrics
    """
    report = {}
    
    # Basic coordinate validation
    nan_indices = detect_nan_coordinates(coordinates)
    inf_indices = detect_inf_coordinates(coordinates)
    outlier_indices = check_coordinate_ranges(coordinates)
    
    report['total_points'] = len(coordinates)
    report['nan_count'] = len(nan_indices)
    report['nan_indices'] = nan_indices
    report['inf_count'] = len(inf_indices)
    report['inf_indices'] = inf_indices
    report['outlier_count'] = len(outlier_indices)
    report['outlier_indices'] = outlier_indices
    
    # Chain continuity
    if residue_numbers is not None:
        breaks = detect_chain_breaks(residue_numbers)
        report['chain_breaks'] = len(breaks)
        report['chain_break_indices'] = breaks
        
        segments = segment_chain(residue_numbers)
        report['continuous_segments'] = len(segments)
        report['segment_lengths'] = [len(seg) for seg in segments]
    
    # CA-CA distances
    if len(coordinates) > 1:
        ca_ca_distances = calculate_ca_ca_distances(coordinates)
        short_distances = detect_short_ca_ca_distances(ca_ca_distances)
        long_distances = detect_long_ca_ca_distances(ca_ca_distances)
        
        report['ca_ca_distances'] = ca_ca_distances.tolist()
        report['short_distances_count'] = len(short_distances)
        report['short_distance_indices'] = short_distances
        report['long_distances_count'] = len(long_distances)
        report['long_distance_indices'] = long_distances
        report['mean_ca_ca_distance'] = float(np.mean(ca_ca_distances))
    
    # Summary metrics
    total_issues = (len(nan_indices) + len(inf_indices) + len(outlier_indices) + 
                   (len(breaks) if residue_numbers is not None else 0) + 
                   len(short_distances) + len(long_distances))
    
    report['total_issues'] = total_issues
    report['quality_score'] = (report['total_points'] - total_issues) / report['total_points'] if report['total_points'] > 0 else 0
    
    return report


def count_problematic_residues(quality_report: Dict[str, Any]) -> int:
    """
    Count the number of problematic residues based on quality report.
    
    Args:
        quality_report: Dictionary containing quality metrics
        
    Returns:
        Count of problematic residues
    """
    # Combine all problematic indices from the report
    problematic_indices = set()
    
    if 'nan_indices' in quality_report:
        problematic_indices.update(quality_report['nan_indices'])
    
    if 'inf_indices' in quality_report:
        problematic_indices.update(quality_report['inf_indices'])
    
    if 'outlier_indices' in quality_report:
        problematic_indices.update(quality_report['outlier_indices'])
    
    # For distances, problematic residues would be the pairs of consecutive residues
    if 'short_distance_indices' in quality_report:
        for idx in quality_report['short_distance_indices']:
            problematic_indices.add(idx)
            problematic_indices.add(idx + 1)
    
    if 'long_distance_indices' in quality_report:
        for idx in quality_report['long_distance_indices']:
            problematic_indices.add(idx)
            problematic_indices.add(idx + 1)
    
    return len(problematic_indices)


def log_corrections(corrections: List[str]) -> None:
    """
    Log all corrections made during validation.
    
    Args:
        corrections: List of correction messages
    """
    for correction in corrections:
        print(f"LOG: {correction}")


def validate_coordinates(coordinates: np.ndarray) -> Dict[str, Any]:
    """
    Main validation function for coordinates.
    
    Args:
        coordinates: Array of shape (n, 3) with coordinates
        
    Returns:
        Dictionary with validation results
    """
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("Coordinates must be a 2D array with shape (n, 3)")
    
    # Generate a basic quality report
    report = generate_quality_report(coordinates)
    
    # Identify issues
    issues = []
    
    if report['nan_count'] > 0:
        issues.append(f"Found {report['nan_count']} NaN coordinates")
    
    if report['inf_count'] > 0:
        issues.append(f"Found {report['inf_count']} infinite coordinates")
    
    if report['outlier_count'] > 0:
        issues.append(f"Found {report['outlier_count']} outlier coordinates")
    
    if 'short_distances_count' in report and report['short_distances_count'] > 0:
        issues.append(f"Found {report['short_distances_count']} short CA-CA distances")
    
    if 'long_distances_count' in report and report['long_distances_count'] > 0:
        issues.append(f"Found {report['long_distances_count']} long CA-CA distances")
    
    report['issues'] = issues
    
    return report
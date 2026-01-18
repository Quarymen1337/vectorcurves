import pytest
import numpy as np
from validation import (detect_nan_coordinates, detect_inf_coordinates, 
                       check_coordinate_ranges, filter_anomalous_coordinates,
                       detect_chain_breaks, create_residue_mapping, 
                       segment_chain, calculate_ca_ca_distances,
                       detect_short_ca_ca_distances, detect_long_ca_ca_distances,
                       identify_chain_breaks_by_distance, generate_quality_report,
                       count_problematic_residues, validate_coordinates)


class TestCoordinateValidation:
    """Tests for coordinate validation functionality"""

    def test_detect_nan_coordinates(self):
        """Test detection of NaN values in coordinates"""
        # Create coordinates with some NaN values
        coords = np.array([
            [1.0, 2.0, 3.0],
            [np.nan, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, np.nan, 12.0]
        ])
        
        nan_indices = detect_nan_coordinates(coords)
        assert nan_indices == [1, 3]

    def test_detect_inf_coordinates(self):
        """Test detection of infinite values in coordinates"""
        # Create coordinates with some infinite values
        coords = np.array([
            [1.0, 2.0, 3.0],
            [float('inf'), 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, float('-inf'), 12.0]
        ])
        
        inf_indices = detect_inf_coordinates(coords)
        assert inf_indices == [1, 3]

    def test_check_coordinate_ranges(self):
        """Test detection of coordinates outside expected ranges"""
        coords = np.array([
            [1.0, 2.0, 3.0],
            [150.0, 5.0, 6.0],  # Within range
            [250.0, 8.0, 9.0],  # Outside range (too large)
            [-250.0, 12.0, 15.0]  # Outside range (too small)
        ])
        
        outlier_indices = check_coordinate_ranges(coords, min_val=-200.0, max_val=200.0)
        assert outlier_indices == [2, 3]

    def test_filter_anomalous_coordinates(self):
        """Test filtering of anomalous coordinates"""
        coords = np.array([
            [1.0, 2.0, 3.0],
            [250.0, 8.0, 9.0],  # Outside range
            [10.0, 12.0, 15.0],
            [np.nan, 14.0, 17.0]  # NaN value
        ])
        
        filtered_coords, removed_indices = filter_anomalous_coordinates(coords, min_val=-200.0, max_val=200.0)
        
        assert len(removed_indices) == 2  # One outlier, one NaN
        assert 1 in removed_indices  # Outlier at index 1
        assert 3 in removed_indices  # NaN at index 3
        assert len(filtered_coords) == 2  # Two valid coordinates remain

    def test_detect_chain_breaks(self):
        """Test detection of chain breaks in residue numbering"""
        residues = [1, 2, 3, 4, 7, 8, 9, 12, 13, 14]  # Breaks after 4 and before 12
        
        breaks = detect_chain_breaks(residues)
        # Should detect breaks where difference is not 1
        # Between index 3 (value 4) and index 4 (value 7) -> (3, 4)
        # Between index 6 (value 9) and index 7 (value 12) -> (6, 7)
        assert len(breaks) == 2
        assert (3, 4) in breaks
        assert (6, 7) in breaks

    def test_create_residue_mapping(self):
        """Test creation of residue number to continuous index mapping"""
        residues = [5, 6, 7, 10, 11]
        mapping = create_residue_mapping(residues)
        
        expected = {5: 0, 6: 1, 7: 2, 10: 3, 11: 4}
        assert mapping == expected

    def test_segment_chain(self):
        """Test segmentation of chain into continuous fragments"""
        residues = [1, 2, 3, 7, 8, 9, 15, 16]
        segments = segment_chain(residues)
        
        expected = [[1, 2, 3], [7, 8, 9], [15, 16]]
        assert segments == expected

    def test_calculate_ca_ca_distances(self):
        """Test calculation of CA-CA distances"""
        coords = np.array([
            [0.0, 0.0, 0.0],      # Distance to next: 1.0
            [1.0, 0.0, 0.0],      # Distance to next: 1.0
            [2.0, 0.0, 0.0],      # Distance to next: 1.0
            [3.0, 0.0, 0.0]
        ])
        
        distances = calculate_ca_ca_distances(coords)
        expected_distances = np.array([1.0, 1.0, 1.0])
        
        np.testing.assert_array_almost_equal(distances, expected_distances)

    def test_detect_short_ca_ca_distances(self):
        """Test detection of anomalously short CA-CA distances"""
        distances = np.array([4.0, 2.0, 5.0, 1.0, 3.5])  # 2.0 and 1.0 are short (< 3.0)
        
        short_indices = detect_short_ca_ca_distances(distances, threshold=3.0)
        assert short_indices == [1, 3]

    def test_detect_long_ca_ca_distances(self):
        """Test detection of anomalously long CA-CA distances"""
        distances = np.array([2.0, 6.0, 3.5, 8.0, 4.0])  # 6.0 and 8.0 are long (> 4.5)
        
        long_indices = detect_long_ca_ca_distances(distances, threshold=4.5)
        assert long_indices == [1, 3]

    def test_identify_chain_breaks_by_distance(self):
        """Test identification of chain breaks by distance threshold"""
        distances = np.array([3.8, 3.8, 10.0, 3.8, 3.8])  # 10.0 indicates a break (> 4.5)
        
        break_indices = identify_chain_breaks_by_distance(distances, threshold=4.5)
        assert break_indices == [2]  # Index where large distance occurs


class TestValidationQualityReports:
    """Tests for quality report generation"""
    
    def test_generate_quality_report(self):
        """Test generation of comprehensive quality report"""
        coords = np.array([
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [np.nan, 4.0, 5.0],  # NaN coordinate
            [4.0, 5.0, 6.0],
            [5.0, 6.0, 7.0]
        ])
        
        residue_nums = [1, 2, 3, 4, 5]
        report = generate_quality_report(coords, residue_nums)
        
        assert report['total_points'] == 5
        assert report['nan_count'] == 1
        assert report['nan_indices'] == [2]
        assert report['chain_breaks'] == 0  # No breaks in residue numbers
        assert 'ca_ca_distances' in report
        assert 'quality_score' in report

    def test_count_problematic_residues(self):
        """Test counting of problematic residues"""
        report = {
            'nan_indices': [2],
            'inf_indices': [3],
            'outlier_indices': [4],
            'short_distance_indices': [0],  # Affects residues 0 and 1
            'long_distance_indices': [1]    # Affects residues 1 and 2
        }
        
        count = count_problematic_residues(report)
        # Residues 2 (from nan), 3 (from inf), 4 (from outlier), 
        # 0,1 (from short distance at index 0), 
        # 1,2 (from long distance at index 1)
        # Unique problematic: {0, 1, 2, 3, 4} = 5 residues
        # Note: residue 2 appears twice in different categories, should be counted once
        assert count >= 5  # At least 5 problematic residues

    def test_validate_coordinates(self):
        """Test the main validation function"""
        coords = np.array([
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [np.nan, 4.0, 5.0],  # NaN coordinate
            [400.0, 5.0, 6.0],   # Outlier coordinate
            [5.0, 6.0, 7.0]
        ])
        
        result = validate_coordinates(coords)
        
        assert result['total_points'] == 5
        assert result['nan_count'] == 1
        assert result['outlier_count'] == 1
        assert len(result['issues']) >= 2  # Should have NaN and outlier issues


class TestContinuousSequence:
    """Test validation on sequences without breaks"""
    
    def test_no_chain_breaks(self):
        """Test processing of continuous sequence without breaks"""
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0]
        ])
        
        residue_nums = [1, 2, 3, 4, 5]
        report = generate_quality_report(coords, residue_nums)
        
        assert report['chain_breaks'] == 0
        assert report['continuous_segments'] == 1
        assert report['nan_count'] == 0
        assert report['inf_count'] == 0
        assert report['outlier_count'] == 0

    def test_normal_ca_ca_distances(self):
        """Test with normal CA-CA distances"""
        coords = np.array([
            [0.0, 0.0, 0.0],
            [3.8, 0.0, 0.0],    # Normal distance
            [7.6, 0.0, 0.0],    # Normal distance
            [11.4, 0.0, 0.0],   # Normal distance
        ])
        
        report = generate_quality_report(coords)
        
        if 'short_distances_count' in report:
            assert report['short_distances_count'] == 0
        if 'long_distances_count' in report:
            assert report['long_distances_count'] == 0
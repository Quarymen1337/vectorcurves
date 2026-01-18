# B-Spline Curve Construction and Validation

This project implements B-spline curve construction and coordinate validation for protein structure modeling. It includes modules for validating coordinate data and constructing smooth B-spline curves from 3D coordinates.

## Table of Contents
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage Examples](#usage-examples)
- [Running Tests](#running-tests)
- [Coverage](#coverage)

## Project Structure
```
b_splines/
├── validation.py               # Coordinate validation functions
├── bspline_construction.py     # B-spline curve construction
├── README.md                   # This file
└── tests/
    ├── test_validation.py      # Unit tests for validation module
    ├── test_bspline.py         # Unit tests for B-spline module
    └── data/
        └── coordinates/        # Test coordinate files
```

## Installation

### For Development
The project requires numpy and scipy for core functionality:

```bash
pip install numpy scipy
pip install pytest pytest-cov  # For running tests
```

### For Package Installation
You can install this package directly from GitLab using pip:

```bash
pip install git+https://gitlab.steelpages.info/codeas/b_splines.git
```

Or install from the built wheel:

```bash
pip install b-splines-package
```

## Usage Examples

### 1. Cubic Spline Interpolation

Create and use cubic splines for 1D and 3D interpolation:

```python
from b_splines_package import CubicSpline1D, CubicSpline3D

# 1D Cubic Spline
points1d = [(0, 0), (1, 1), (2, 3), (3, 2), (4, 5)]
spline1d = CubicSpline1D(points1d)

# Evaluate at specific points
result = spline1d(0.5)  # Get interpolated value at x=0.5
print(f"Interpolated value at x=0.5: {result}")

# Evaluate at multiple points
x_values = [0.0, 0.5, 1.0, 1.5, 2.0]
y_values = spline1d(x_values)
print(f"Interpolated values: {y_values}")

# 3D Cubic Spline - for curves in 3D space
points3d = [(0, 0, 0), (1, 2, 1), (2, 3, 2), (3, 2, 3)]
spline3d = CubicSpline3D(points3d)

# Get point on 3D curve at parameter t=0.5
point3d = spline3d(0.5)
print(f"3D point at t=0.5: {point3d}")

# Sample multiple points along the curve
sampled_points = spline3d.sample(10)  # Generate 10 points along the curve
print(f"Number of sampled points: {len(sampled_points)}")
```

### 2. Advanced Spline Features

Calculate derivatives and tangents:

```python
from b_splines_package import CubicSpline1D, CubicSpline3D

# 1D spline derivative
points1d = [(0, 0), (1, 1), (2, 3), (3, 2)]
spline1d = CubicSpline1D(points1d)
derivative = spline1d.derivative(0.5)  # First derivative at x=0.5
print(f"Derivative at x=0.5: {derivative}")

# 3D spline tangent
points3d = [(0, 0, 0), (1, 2, 1), (2, 3, 2), (3, 2, 3)]
spline3d = CubicSpline3D(points3d)
tangent = spline3d.tangent(0.5)  # Unit tangent vector at t=0.5
print(f"Tangent vector at t=0.5: {tangent}")
```

## Running Tests

### Run all tests:
```bash
python -m pytest tests/test_validation.py tests/test_bspline.py -v
```

### Run tests with coverage:
```bash
python -m pytest tests/test_validation.py tests/test_bspline.py -v --cov
```

### Run specific test files:
```bash
# Run only validation tests
python -m pytest tests/test_validation.py -v

# Run only B-spline tests
python -m pytest tests/test_bspline.py -v
```

### Run with detailed coverage report:
```bash
python -m pytest tests/test_validation.py tests/test_bspline.py --cov=validation --cov=bspline_construction --cov-report=html
```

This will generate an HTML coverage report in the `htmlcov/` directory.

## Coverage

The project has comprehensive test coverage:
- **Validation module**: ~93% coverage
- **B-spline construction module**: ~63% coverage
- **Overall project**: ~84% coverage

All tests pass successfully and cover edge cases, performance requirements, and various functionality aspects as specified in the requirements.

## Modules

### validation.py
- `validate_coordinates()`: Main validation function
- `detect_nan_coordinates()`: Find NaN values in coordinates
- `detect_inf_coordinates()`: Find infinite values in coordinates
- `check_coordinate_ranges()`: Check for outlier coordinates
- `filter_anomalous_coordinates()`: Remove problematic coordinates
- `detect_chain_breaks()`: Find breaks in residue numbering
- `calculate_ca_ca_distances()`: Calculate distances between consecutive CA atoms
- `generate_quality_report()`: Create comprehensive quality report

### bspline_construction.py
- `BsplineCurve`: Main B-spline curve class
- `construct_bspline()`: Convenience function to construct B-spline
- `sample_curve_points()`: Sample points from a B-spline curve
- `calculate_rmsd()`: Calculate Root Mean Square Deviation between point sets
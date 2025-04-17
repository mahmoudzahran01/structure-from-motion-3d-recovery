# Affine Structure from Motion

This repository contains an implementation of the Affine Structure from Motion algorithm, which recovers a 3D point cloud from a sequence of images.

## Overview

Affine Structure from Motion (SfM) is a computer vision technique that reconstructs 3D information from 2D image sequences. This implementation:

1. Takes tracked 2D points across multiple frames as input
2. Creates a measurement matrix by normalizing the points
3. Applies singular value decomposition (SVD) to factorize the matrix
4. Enforces orthographic constraints to obtain 3D structure
5. Provides visualization tools for the reconstructed 3D point cloud and camera positions

## Example Results

![3D Point Cloud](images/point_cloud.png)

The algorithm reconstructs the 3D structure of a scene from 2D point correspondences tracked across multiple frames.

## Usage

```python
# Load your track data
import numpy as np
from sfm import affineSFM, plot_X, plot_cam_pos

# Example with tracked points from multiple frames
track_x = your_x_coordinates  # Shape: (num_frames, num_points)
track_y = your_y_coordinates  # Shape: (num_frames, num_points)

# Remove rows with NaN values
is_nan = np.isnan(track_x) | np.isnan(track_y)
is_valid = ~is_nan
valid_row_indices = is_valid.all(axis=1)
track_x, track_y = track_x[valid_row_indices], track_y[valid_row_indices]

# Run SfM algorithm
A, X = affineSFM(track_x, track_y)

# Visualize results
plot_X(X)  # Plot 3D point cloud
plot_cam_pos(A)  # Plot camera positions
```

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- SciPy (for loading .mat files)

## Implementation Details

The implementation follows these key steps:

1. **Measurement Matrix Creation**: Normalizes track points to zero mean and creates the measurement matrix
2. **Matrix Factorization**: Uses SVD to decompose the measurement matrix and enforces rank 3 constraint
3. **Orthographic Constraint Application**: Solves for the appropriate transformation matrix
4. **Structure Reconstruction**: Recovers the 3D coordinates and camera motion

## License

MIT License
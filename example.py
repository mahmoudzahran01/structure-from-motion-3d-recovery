"""
Example script demonstrating the usage of the Affine Structure from Motion algorithm.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from sfm import affineSFM, plot_X, plot_cam_pos

def main():
    """
    Run a demo of the Affine Structure from Motion algorithm.
    """
    # Check if data exists, if not, provide download instructions
    if not os.path.exists('data/tracks.mat'):
        print("Data file not found. Please download the tracks.mat file and place it in the data/ directory.")
        print("You can find the data at: https://drive.google.com/file/d/1A0Rin_YMmWkExjI99vfLYvU_dy-9gFTT/view")
        return
    
    # Load track data
    data = loadmat('data/tracks.mat')
    track_x = data['track_x']
    track_y = data['track_y']
    
    # Check for NaN values in the tracks
    is_nan = np.isnan(track_x) | np.isnan(track_y)
    num_points = np.prod(track_x.shape)
    nan_count = is_nan.sum()
    print(f"Total points: {num_points}, NaN count: {nan_count}")
    
    # Remove rows with NaN values
    is_valid = ~is_nan
    valid_row_indices = is_valid.all(axis=1)
    track_x_clean = track_x[valid_row_indices]
    track_y_clean = track_y[valid_row_indices]
    print(f"Using {track_x_clean.shape[0]} valid frames out of {track_x.shape[0]} total frames")
    
    # Run SfM algorithm
    print("Running Affine Structure from Motion algorithm...")
    A, X = affineSFM(track_x_clean, track_y_clean)
    
    # Save results directory
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Plot 3D point cloud with different viewpoints
    print("Generating visualizations...")
    
    # Default viewpoint
    plot_X(X)
    
    # Alternate viewpoints
    plot_X(X, elev=-90, azim=-50)
    plot_X(X, elev=-110, azim=-60)
    
    # Plot camera positions
    plot_cam_pos(A)
    
    print("Done!")

if __name__ == "__main__":
    main()
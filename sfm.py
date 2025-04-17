"""
Affine Structure from Motion (SfM) implementation.

This module implements the Affine SfM algorithm to recover 3D structure from
2D tracked points across multiple frames.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_measurement_matrix(x, y):
    """
    Create measurement matrix from tracked points.
    
    Args:
        x: x-coordinates of tracked points, shape (num_frames, num_points)
        y: y-coordinates of tracked points, shape (num_frames, num_points)
        
    Returns:
        D_mat: Measurement matrix, shape (2*num_frames, num_points)
    """
    x_mean, y_mean = x.mean(axis=0, keepdims=True), y.mean(axis=0, keepdims=True)
    x, y = x-x_mean, y-y_mean
    D_mat = np.vstack((x.T, y.T))
    return D_mat

def affineSFM(x, y):
    """
    Affine structure from motion algorithm.
    
    Args:
        x: x-coordinates of tracked points, shape (num_frames, num_points)
        y: y-coordinates of tracked points, shape (num_frames, num_points)
        
    Returns:
        A: Camera motion matrix
        X: 3D coordinates of points
    """
    # Create measurement matrix
    D = get_measurement_matrix(x, y)
    
    # Decompose and enforce rank 3
    U, W, Vt = np.linalg.svd(D)
    
    W = np.diag(W)
    U_3 = U[:, :3]
    Vt_3 = Vt[:3, :]
    W_3 = W[:3, :3]
    
    # Factor the measurement matrix
    A_telda = np.dot(U_3, W_3**(1/2))
    X_telda = np.dot(W_3**(1/2), Vt_3)
    
    # Apply orthographic constraints
    A_telda_num_rows = len(A_telda)
    m = A_telda_num_rows//2
    
    a_telda_i1_t = A_telda[:m, :]
    a_telda_i2_t = A_telda[m:, :]
    
    perform_outer = lambda v1, v2: np.outer(v1, v2).flatten()
    
    M = np.array([np.vstack((perform_outer(a_telda_i1_t[i], a_telda_i1_t[i]),
                             perform_outer(a_telda_i2_t[i], a_telda_i2_t[i]),
                             perform_outer(a_telda_i2_t[i], a_telda_i1_t[i]))) 
                  for i in range(m)]).reshape(-1, 9)
    
    k = np.array([1, 1, 0]).reshape(-1, 1)
    k = np.tile(k, (m, 1))
    
    l = np.dot(np.linalg.pinv(M), k)
    L = l.reshape(3, 3)
    
    # Find correct transformation
    C = np.linalg.cholesky(L)
    A = np.dot(A_telda, C)
    X = np.dot(np.linalg.inv(C), X_telda)
    
    return A, X

def plot_X(X, elev=-110, azim=10):
    """
    Plot the 3D point cloud.
    
    Args:
        X: 3D coordinates of points, shape (3, num_points)
        elev: Elevation angle for 3D plot viewing
        azim: Azimuthal angle for 3D plot viewing
    """
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(projection='3d')
    xs, ys, zs = X
    ax.scatter(xs, ys, zs, color='r')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    ax.set_title(f'3D Point Cloud (Elevation: {elev}, Azimuth: {azim})')
    ax.view_init(elev, azim)
    plt.show()

def get_cam_pos(A):
    """
    Calculate camera positions from motion matrix.
    
    Args:
        A: Camera motion matrix
        
    Returns:
        camera_pos: Camera positions, shape (num_frames, 3)
    """
    A_num_rows = len(A)
    m = A_num_rows//2
    camera_pos = np.zeros((m, 3))
    
    for i in range(m):
        cross = np.cross(A[i, :], A[i+m, :])
        normalized_cross = cross/np.linalg.norm(cross)
        camera_pos[i, :] = normalized_cross
    
    return camera_pos

def plot_cam_pos(A):
    """
    Plot camera positions over frames.
    
    Args:
        A: Camera motion matrix
    """
    cam_pos = get_cam_pos(A)
    
    plt.figure(figsize=(9, 9))
    
    plt.subplot(3, 1, 1)
    plt.plot(cam_pos[:, 0])
    plt.xlabel('Frame')
    plt.ylabel('X Position')
    plt.title('Camera Position (X)')
    
    plt.subplot(3, 1, 2)
    plt.plot(cam_pos[:, 1])
    plt.xlabel('Frame')
    plt.ylabel('Y Position')
    plt.title('Camera Position (Y)')
    
    plt.subplot(3, 1, 3)
    plt.plot(cam_pos[:, 2])
    plt.xlabel('Frame')
    plt.ylabel('Z Position')
    plt.title('Camera Position (Z)')
    
    plt.tight_layout()
    plt.show()
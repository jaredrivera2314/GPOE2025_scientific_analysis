import os
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from skimage import measure
import h5py
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

from ART_utils import *


# Main execution
if __name__ == "__main__":
    image_source = 'h5'  # Set True for synthetic, to False to use local images, and "h5" to pull from h5
    zeros = False # Set to False to use spherical initial guess
    folder_path = "C:/Users/jared/OneDrive/Desktop/GPOE2025/build day data/"  # Replace with your folder path
    timestamp = datetime.strptime("2023-12-25 10:30:00", "%Y-%m-%d %H:%M:%S")

    # Parameters
    alpha = 0.1  # Step size for ART
    num_iterations = 0  # Number of iterations
    voxel_size = 20000  # Physical size of the voxels
    num_voxels = 12  # Number of voxels in each dimension

    # Camera intrinsics (example values, adjust as needed)
    focal_length = 1.0
    camera_matrix = np.array([[focal_length, 0, voxel_size / 2],
                           [0, focal_length, voxel_size / 2],
                           [0, 0, 1]])
    
    # if not from h5
    camera_positions = [
        [-100000, 60000, 0, -3.6*np.pi/8, 0],  # Camera 1
        [100000, 60000, 0, 1.4*3*np.pi/4, 0],  # Camera 2 
        [0, -100000, 0, np.pi/4, 0],  # Camera 3 
        ]
    camera_count = len(camera_positions)  # Number of cameras

    # if synthetic, define test sphere
    synth_flag=0
    if image_source == 'synthetic':
        synth_flag=1
    sphere_radius = 10000
    sphere_height = 60000

    
    if image_source=='synthetic':
        images = generate_synthetic_images(camera_positions, sphere_radius, sphere_height)
    elif image_source=='folder':
        images = load_images_from_folder(folder_path, camera_count)
    elif image_source=='h5':
        images, camera_positions = load_images_from_h5(folder_path, timestamp)

    if np.size(images)>0:
        # build some ansatz
        v0=initialize_volume_with_sphere(num_voxels, voxel_size, sphere_radius, sphere_height)
        volume = reconstruct(images,v0,num_iterations,voxel_size)
        # For debugging, show projected images alongside their partner real image
        visualize_volume(volume, camera_positions,sphere_radius, sphere_height,voxel_size,synth_flag)
    else:
        print("No images generated or found.")

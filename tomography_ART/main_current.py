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
    # DO 28TH 2AM, wormsloth, camel, capybarapus, llama-goose, axolata
    folder_path = "C:/Users/jared/OneDrive/Desktop/GPOE2025/all-sky camera/Data/"  # Replace with your folder path 
    timestamp = datetime.strptime("2023-12-25 10:30:00", "%Y-%m-%d %H:%M:%S")

    # Parameters
    alpha = 0.05  # Step size for ART, 0.1?
    num_iterations = 1  # Number of iterations
    voxel_size = 20000  # Physical size of the voxels
    num_voxels = 12  # Number of voxels in each dimension

    # Camera intrinsics (example values, adjust as needed)
    focal_length = 1.56e-3 #1.56mm
    pixel_size = 1.55e-6 #1.55um pixels
    camera_matrix = np.array([[focal_length/pixel_size, 0, 750], #cx?
                           [0, focal_length/pixel_size, 750], #cy?
                           [0, 0, 1]])
    
    # if not from h5
    camera_positions = [
        [-100000, 60000, 0, -3.6*np.pi/8, 0],  # Camera 1
        [100000, 60000, 0, 1.4*3*np.pi/4, 0],  # Camera 2 
        [0, -100000, 0, np.pi/4, 0],  # Camera 3 pi/4,0
        ]
    camera_count = len(camera_positions)  # Number of cameras

    # if synthetic, define test sphere
    synth_flag=0
    if image_source == 'synthetic':
        synth_flag=1
    sphere_radius = 50000
    sphere_position = np.array([0,0,150000])

    
    if image_source=='synthetic':
        images = generate_synthetic_images(camera_positions, sphere_radius, sphere_position,num_voxels,voxel_size,camera_matrix)
    elif image_source=='folder':
        images = load_images_from_folder(folder_path, camera_count)
    elif image_source=='h5':
        images, camera_positions = load_images_from_h5(folder_path, timestamp)

    if np.size(images)>0:
        im_size = np.shape(images[0])
        print(im_size)
        # build some ansatz
        v0=initialize_volume_with_sphere(num_voxels, voxel_size, sphere_radius, sphere_position)
        v0=np.zeros((num_voxels, num_voxels, num_voxels))
        volume = reconstruct(images,v0,num_iterations,voxel_size,camera_positions,im_size,camera_matrix,alpha,num_voxels,pixel_size)
        # For debugging, show projected images alongside their partner real image
        visualize_volume(volume, camera_positions,sphere_radius, sphere_position,voxel_size,synth_flag) 
    else:
        print("No images generated or found.")

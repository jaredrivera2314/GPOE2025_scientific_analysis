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



def load_images_from_h5(folder_path, timestamp):
    #point to data folder and specify a time
    #load all images at the given timestamp 
    # filename tells hour before -
    mainpath = folder_path

    # Inititialize arrays
    camera_count = 0
    for entry in os.scandir(mainpath):
        if entry.is_dir():
            camera_count += 1

    exposures=[]
    camera_positions=[]
    # Local datum at Poker
    earth_radius = 6700000
    lat_poker = 44.510719 * np.pi/180
    long_poker = -72.769515 * np.pi/180
    alt_poker = 889

    # Pull date/hour from timestamp

    # Loop through appropriate images
    for i in range(camera_count):
        cam_name = "penguin" # index logic here, may want to save name with exposures/pos somehow
        impath="0-exposures.hdf5" #timestamp-based logic here
        filename = mainpath+cam_name+"/"+impath

        # Rip data
        timestamps = list(h5py.File(filename, "r")['timestamp']) #UTC seconds since Jan1 1970
        data = list(h5py.File(filename, "r")['exposure'])
        exposures.append(data[0][1]) #  Do any cropping here
        df = pd.read_excel(mainpath+'locations.xlsx')
        lat = df['Latitude'][0]
        long = df['Longitude'][0]
        alt = df['Altitude'][0]

        x = earth_radius * (long - long_poker)
        y = earth_radius * (lat - lat_poker)
    
        # Example camera extrinsics: [translation_x, translation_y, translation_z, az_rotation_angle, polar_angle]
        camera_positions.append([x,y,alt - alt_poker,long-long_poker,0])

    return exposures, camera_positions

# Load images from a folder and associate them with camera indices
def load_images_from_folder(folder, camera_count):
    camera_images = {i: [] for i in range(camera_count)}
    for i in range(camera_count):
        camera_folder = os.path.join(folder, f'camera_{i}')
        if os.path.exists(camera_folder):
            for filename in os.listdir(camera_folder):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(camera_folder, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        camera_images[i].append(img)

    return camera_images

def generate_sphere_points(radius, center, num_points=30):

    theta = np.linspace(0, np.pi, num_points)  # Polar angle (0 to pi)
    phi = np.linspace(0, 2 * np.pi, num_points)  # Azimuthal angle (0 to 2*pi)
    
    points = []
    
    for t in theta:
        for p in phi:
            x = radius * np.sin(t) * np.cos(p) + center[0]
            y = radius * np.sin(t) * np.sin(p) + center[1]
            z = radius * np.cos(t) + center[2]
            points.append([x, y, z])
    
    return np.array(points)

def rotation_matrix(azimuth, polar):
    cos_az = np.cos(azimuth)
    sin_az = np.sin(azimuth)
    cos_pol = np.cos(polar)
    sin_pol = np.sin(polar)

    rotation_yaw = np.array([[cos_az, -sin_az, 0],
                              [sin_az, cos_az, 0],
                              [0, 0, 1]])

    rotation_pitch = np.array([[1, 0, 0],
                                [0, cos_pol, -sin_pol],
                                [0, sin_pol, cos_pol]])

    return rotation_pitch @ rotation_yaw


# Function to perform 3D to 2D projection
def project_3d_to_2d(camera_positions, object_points, focal_length=1.0):
    # Camera orientation is a rotation matrix to rotate the world coordinates to camera coordinates
    x_camera, y_camera, z_camera, yaw, pitch = camera_positions
    camera_position = np.array([x_camera, y_camera, z_camera])
    rotation_mat = rotation_matrix(yaw, pitch)
    
    # Translate object points to camera coordinates by subtracting camera position
    object_in_camera_coords = object_points - camera_position
    
    # Project points onto the 2D image plane using the pinhole model
    projected_points = []
    for point in object_in_camera_coords:
        # Apply the camera rotation (rotate the point in world coordinates to camera space)
        rotated_point = rotation_mat @ point
        # Apply the pinhole camera projection: simple perspective projection
        x_image = (focal_length * rotated_point[0]) / rotated_point[2]  # x = focal_length * X / Z
        y_image = (focal_length * rotated_point[1]) / rotated_point[2]  # y = focal_length * Y / Z
        projected_points.append([x_image, y_image])
    
    return np.array(projected_points)

# Generate synthetic images simulating a sphere
def generate_synthetic_images(camera_positions, radius, height):
    synthetic_images = {i: [] for i in range(len(camera_positions))}
    center = np.array([0, 0, height])
    sphere_points = generate_sphere_points(radius, center)

    for i in range(len(camera_positions)):
        synthetic_images[i] = project_3d_to_2d(camera_positions[i],sphere_points)
    
    return synthetic_images

# Initialize 3D volume (voxel grid)
def initialize_volume_with_sphere(size, voxelsize,radius, height):
    volume = np.zeros((size, size, size), dtype=np.float32)

    # Create a sphere within the volume
    for x in range(size):
        for y in range(size):
            for z in range(size):
                # Calculate the distance from the center
                distance = np.sqrt((voxelsize*x)**2 + (voxelsize*y)**2 + (voxelsize*z - height)**2)
                if distance < radius:
                    volume[x, y, z] = 1.0  # Set the voxel to 1 (inside the sphere)

    return volume

# Projection function
def project(volume, position, azimuth, polar):
    rotation = rotation_matrix(azimuth, polar)
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation
    transformation_matrix[:3, 3] = position

    projected_image = np.zeros((num_voxels, num_voxels), dtype=np.float32)

    for x in range(volume.shape[0]):
        for y in range(volume.shape[1]):
            for z in range(volume.shape[2]):
                # Scale voxel position by voxel_size
                point_3d = np.array([x * voxel_size, y * voxel_size, z * voxel_size, 1])
                transformed_point = transformation_matrix @ point_3d

                if transformed_point[2] > 0:  # Avoid division by zero
                    # Project 3D point to 2D image plane
                    x_proj = int((transformed_point[0] / transformed_point[2]) * camera_matrix[0, 0] + camera_matrix[0, 2])
                    y_proj = int((transformed_point[1] / transformed_point[2]) * camera_matrix[1, 1] + camera_matrix[1, 2])

                    if 0 <= x_proj < projected_image.shape[1] and 0 <= y_proj < projected_image.shape[0]:
                        projected_image[y_proj, x_proj] += volume[x, y, z]

    return projected_image

# Update function based on ART
def update_volume(volume, observed, projected):
    observed_resized = cv2.resize(observed, (num_voxels, num_voxels))
    residual = observed_resized.astype(np.float32) - projected.astype(np.float32)

    # Debug: Check values of residual
    # print("Residual values:", residual)

    residual = np.clip(residual, -255, 255)
    volume += alpha * residual[:, :, np.newaxis]
    volume = np.clip(volume, 0, 255)

    return volume

# Main reconstruction function
def reconstruct(camera_images, ansatz, num_iterations, voxel_size):
    volume = ansatz

    for i in range(num_iterations):
        for cam_index, images in camera_images.items():
            for img in images:
                resized_img = cv2.resize(img, (num_voxels, num_voxels))
                projected = project(volume, camera_positions[cam_index][:3], camera_positions[cam_index][3], camera_positions[cam_index][4])

                # Debug: Visualize projected images and volume updates
                if i % 5 == 10:  # Every 5 iterations
                    plt.imshow(projected, cmap='gray')
                    plt.title(f'Projected Image from Camera {cam_index} - Iteration {i}')
                    plt.show()
                    print(f"Volume after iteration {i}:\n", volume)

                volume = update_volume(volume, resized_img, projected)

    return volume


def visualize_volume(volume, camera_positions,sphere_radius,sphere_height, voxel_size, synth_flag):
    scale_factor = voxel_size / 2  # Scale vertices for physical size
    volume = volume * scale_factor
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(volume[:,0],volume[:,1],volume[:,2],color='cyan', linewidth=0.1, label='Reconstructed Volume')
    
    if synth_flag:
        # Add synthetic sphere to the visualization
        sphere_points = generate_sphere_points(sphere_radius, np.array([0,0,sphere_height]))
        sphere_x, sphere_y, sphere_z = sphere_points[:,0], sphere_points[:,1], sphere_points[:,2]

        ax.scatter(sphere_x, sphere_y, sphere_z, color='red', alpha=0.05, label='Synthetic Sphere', s=1)
        ax.set_title('Reconstructed 3D Object with Synthetic Sphere')
    else:
        ax.set_title('Reconstructed 3D Object')

    def draw_camera_fov(ax, position, azimuth, polar):
        fov_radius = 2 * scale_factor
        fov_vectors = np.array([[fov_radius, 0, 0], [0, fov_radius, 0], [0, 0, -fov_radius]])
        rotation = rotation_matrix(azimuth, polar)
        rotated_vectors = fov_vectors @ rotation.T
        fov_corners = position[:3] + rotated_vectors

        for corner in fov_corners:
            ax.plot([position[0], corner[0]], [position[1], corner[1]], [position[2], corner[2]], color='orange')

    for i in range(len(camera_positions)):
        ax.scatter(camera_positions[i][0], camera_positions[i][1], camera_positions[i][2], marker='.', label="Camera " + str(i + 1))
        draw_camera_fov(ax, camera_positions[i][:3], camera_positions[i][3], camera_positions[i][4])

    ax.set_xlabel('X (Physical Size)')
    ax.set_ylabel('Y (Physical Size)')
    ax.set_zlabel('Z (Physical Size)')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
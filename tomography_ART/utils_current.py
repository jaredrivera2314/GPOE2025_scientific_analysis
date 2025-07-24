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
from scipy.ndimage import gaussian_filter



def load_images_from_h5(folder_path, timestamp):
    #point to data folder and specify a time
    #load all images at the given timestamp 
    # filename tells hour before -
    mainpath = folder_path

    # Inititialize arrays
    date_path = '2025-01-28'
    camera_count = 0
    cam_names = []
    for entry in os.scandir(mainpath+"/"+date_path):
        if entry.is_dir():
            camera_count += 1
            cam_names.append(entry.name)

    print(cam_names)
    print(camera_count)
    exposures=[]
    camera_positions=[]
    # Local datum at Poker
    earth_radius = 6378137
    lat_poker = 65.1256 * np.pi/180
    long_poker = -147.4919 * np.pi/180
    alt_poker = 889

    # Pull date/hour from timestamp

    # Loop through appropriate images
    for i in range(camera_count):
        cam_name = cam_names[i] # index logic here, may want to save name with exposures/pos somehow
        impath="2-exposures.hdf5" #timestamp-based logic here (look at shot ~110 below rather than 0 for aurora!!!)
        filename = mainpath+date_path+"/"+cam_name+"/"+impath

        # Rip data
        timestamps = list(h5py.File(filename, "r")['timestamp']) #UTC seconds since Jan1 1970
        data = list(h5py.File(filename, "r")['exposure'])
        exposures.append(data[110][:,:,1][1000:2500,1000:2500].swapaxes(-2,-1)[...,::-1]) #  Do any cropping here ([1000:2500,1000:2500] or so), img.swapaxes(-2,-1)[...,::-1] to rotate
        plt.imshow(data[110][:,:,1][1000:2500,1000:2500])
        plt.title('Image')
        plt.show()
        df = pd.read_excel(mainpath+'Allsky Deployment Locations.xlsx') # throw this csv into GPOEpostprocess
        index = df.loc[df['cam_name'] == cam_name].index #??
        lat = df['latitude'][i]* np.pi/180
        long = df['longitude'][i]* np.pi/180
        alt = df['altitude_m'][i]

        x = earth_radius * (long - long_poker)
        y = earth_radius * (lat - lat_poker)
    
        # Example camera extrinsics: [translation_x, translation_y, translation_z, az_rotation_angle, polar_angle]
        # print([x,y,alt - alt_poker,long-long_poker,0])
        # camera_positions.append([x,y,alt - alt_poker,long-long_poker,0])
        camera_positions.append([x,y,alt - alt_poker,np.pi/2,0])

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

def generate_sphere_points(radius, center, num_points=60):

    theta = np.linspace(0, np.pi, num_points)  # Polar angle (0 to pi)
    phi = np.linspace(0, 2 * np.pi, num_points)  # Azimuthal angle (0 to 2*pi)
    
    points = []
    
    for t in theta:
        for p in phi:
            x = 3*radius * np.sin(t) * np.cos(p) + center[0]
            y = 1/3*radius * np.sin(t) * np.sin(p) + center[1]
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

# Generate synthetic images simulating a sphere
def generate_synthetic_images(camera_positions, radius, height,num_voxels,voxel_size,camera_matrix):
    synthetic_images = {i: [] for i in range(len(camera_positions))}
    # center = np.array([0, 0, height])
    # sphere_points = generate_sphere_points(radius, center) # MOVING FROM generate_sphere_points TO initialize_volume_with_sphere
    sphere_points = initialize_volume_with_sphere(num_voxels, voxel_size, radius, height)
    im_size = (3040, 3326)# copy our images

    for i in range(len(camera_positions)):
        # UPDATING FROM project_3d_to_2d TO project
        # synthetic_images[i] = cv2.resize(project_3d_to_2d(camera_positions[i],sphere_points), (num_voxels, num_voxels))
        #args (camera_positions, object_points, focal_length=1.0) to (volume,im_size,camera_matrix, position, azimuth, polar)
        synthetic_images[i] = cv2.resize(project(sphere_points,im_size,camera_matrix,camera_positions[i]), (num_voxels, num_voxels))
    
    return synthetic_images

# Initialize 3D volume (voxel grid)
def initialize_volume_with_sphere(size, voxelsize, radius, sphere_position):
    volume = np.zeros((size, size, size), dtype=np.float32)

    # Create a sphere within the volume
    for x in range(size):
        for y in range(size):
            for z in range(size):
                # Calculate the distance from the center
                distance = np.sqrt((x-sphere_position[0]/voxelsize)**2 + (y-sphere_position[1]/voxelsize)**2 + (z -sphere_position[2]/voxelsize)**2) #voxelsize*x ?
                # print(distance)
                if distance < radius/voxelsize:
                    volume[x, y, z] = 0.25  # Set the voxel to 1 (inside the sphere)
                    # print('updated')

    return volume

# 3dto 2d projection function
def project(volume,im_size,camera_matrix, orientation, voxel_size, pixel_size): # AZIMUTH AND POLAR ARE ALREADY IN THE POSITION, REDEFINE AS ORIENTATION

    rotation = rotation_matrix(orientation[3], orientation[4]) #azimuth and polar
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation
    transformation_matrix[:3, 3] = orientation[:3] #xyz
    # print(transformation_matrix)

    projected_image = np.zeros(im_size, dtype=np.float32)

    for x in range(volume.shape[0]):
        for y in range(volume.shape[1]):
            for z in range(volume.shape[2]):
                # Scale voxel position by voxel_size z * voxel_size
                point_3d = np.array([voxel_size*x , voxel_size*y , voxel_size*z , 1])
                transformed_point = transformation_matrix @ point_3d
                # print(point_3d)
                # print(transformed_point)
                # print('Z_c: '+str(transformed_point[2]))

                if transformed_point[2] != 0:  # Avoid division by zero
                    # Project 3D point to 2D image plane
                    x_proj = int(((transformed_point[0] / transformed_point[2]) * camera_matrix[0, 0] + camera_matrix[0, 2]))#//pixel_size)
                    # print('xproj: '+str(x_proj))
                    y_proj = int(((transformed_point[1] / transformed_point[2]) * camera_matrix[1, 1] + camera_matrix[1, 2]))#//pixel_size)
                    # print('yproj: '+str(y_proj))

                    if 0 <= x_proj < projected_image.shape[1] and 0 <= y_proj < projected_image.shape[0]:
                        projected_image[y_proj, x_proj] += volume[x, y, z]

    return projected_image

# Update function based on ART
def update_volume(volume, observed, projected, alpha,num_voxels):
    observed_resized = cv2.resize(observed, (num_voxels, num_voxels))
    projected_resized = cv2.resize(projected, (num_voxels, num_voxels)) # why is this even necessary?
    residual = observed - projected # no pre-resizing
    # plt.imshow(residual)
    # plt.title('residual')
    # plt.show()
    # residual = observed_resized.astype(np.float32) - projected_resized.astype(np.float32)
    residual = cv2.resize(residual, (num_voxels, num_voxels))
    residual = np.clip(residual, -255, 255) #-255,255
    # plt.imshow(residual,cmap='gray',vmin=-255/3,vmax=255/3)
    # plt.title('resized residual')
    # plt.colorbar()
    # plt.show()
    # print('max resized residual: '+str(np.max(residual)))
    volume += alpha * residual[:, :, np.newaxis] # is this being added along the correct direction?
    volume = np.clip(volume, 0, 255)

    return volume

# Main reconstruction function
def reconstruct(camera_images, ansatz, num_iterations, voxel_size,camera_positions,im_size,camera_matrix,alpha,num_voxels, pixel_size):
    volume = ansatz * voxel_size
    print('ART:')

    for i in range(num_iterations):
        print(i)
        for j in range(len(camera_images)):
            # resized_img = cv2.resize(camera_images[j], (num_voxels, num_voxels))
            resized_img = camera_images[j] #temp, not sure we need to downsize...
            resized_img = gaussian_filter(resized_img, sigma=50, truncate=10) #TEST
            projected = project(volume,im_size,camera_matrix, camera_positions[j], voxel_size, pixel_size) #j = cam_index
            projected = gaussian_filter(projected, sigma=50, truncate=10) #TEST

            # Debug: Visualize projected images and volume updates
            if i == 0: pass # Every 5 iterations
                # plt.imshow(projected, cmap='gray',vmin=np.min(projected),vmax=1.5*np.max(projected))
                # plt.title('Projection')
                # plt.show()
                # print(f"Volume after iteration {i}:\n", volume)
                # print('max image: '+str(np.max(resized_img)))
                # print('max projected: '+str(np.max(projected)))

            volume = update_volume(volume, resized_img, projected, alpha,num_voxels)

    return volume

def cuboid_data(pos, voxel_size):

    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the (left, outside, bottom) point
    o = [a - b / 2 for a, b in zip(pos, (voxel_size,voxel_size,voxel_size))]
    # get the length, width, and height
    l, w, h = (voxel_size,voxel_size,voxel_size)
    x = np.array([[o[0], o[0] + l, o[0] + l, o[0], o[0]],  
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  
         [o[0], o[0] + l, o[0] + l, o[0], o[0]]])  
    y = np.array([[o[1], o[1], o[1] + w, o[1] + w, o[1]],  
         [o[1], o[1], o[1] + w, o[1] + w, o[1]],  
         [o[1], o[1], o[1], o[1], o[1]],          
         [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]])   
    z = np.array([[o[2], o[2], o[2], o[2], o[2]],                       
         [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],   
         [o[2], o[2], o[2] + h, o[2] + h, o[2]],               
         [o[2], o[2], o[2] + h, o[2] + h, o[2]]])               
    return np.array(x), np.array(y), np.array(z)

def plotCubeAt(pos,density,ax,voxel_size):
    # Plotting a cube element at position pos
    X, Y, Z = cuboid_data(pos, voxel_size)
    ax.plot_surface(X, Y, Z, color='lightgreen', rstride=1, cstride=1, alpha=density**10) #density/2?

def plotMatrix(matrix, voxel_size,sphere_position,ax):
    # plot a Matrix 
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            for k in range(matrix.shape[2]):
                  density = matrix[i,j,k]/np.max(matrix)
                  if k*voxel_size<100000: density=0
                #   pos=((i-matrix.shape[0]/2)*voxel_size+sphere_position[0],(j-matrix.shape[1]/2)*voxel_size+sphere_position[1],(k-matrix.shape[2]/2)*voxel_size+sphere_position[2])
                  pos=(i+sphere_position[0],j+sphere_position[1],k+sphere_position[2])
                  pos=(i*voxel_size,j*voxel_size,k*voxel_size) #works in sandbox...
                  pos=((i-matrix.shape[0]/2)*voxel_size,(j-matrix.shape[1]/2)*voxel_size,(k-0*matrix.shape[1]/2)*voxel_size)
                  plotCubeAt(pos=pos, density=density,ax=ax,voxel_size=voxel_size) #sphere_height-sphere_radius + k


def visualize_volume(volume, camera_positions,sphere_radius,sphere_position, voxel_size, synth_flag):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plotMatrix(volume, voxel_size,sphere_position, ax) # voxels
    
    if synth_flag:
        # Add synthetic sphere to the visualization
        sphere_points = generate_sphere_points(sphere_radius, sphere_position)
        sphere_x, sphere_y, sphere_z = sphere_points[:,0], sphere_points[:,1], sphere_points[:,2]

        ax.scatter(sphere_x, sphere_y, sphere_z, color='green', alpha=0.1, label='Synthetic Sphere', s=1)
        ax.set_title('Reconstructed 3D Object with Synthetic Sphere')
    else:
        ax.set_title('Reconstructed 3D Object')

    def draw_camera_fov(ax, position, azimuth, polar):
        fov_radius = 20000
        fov_vectors = np.array([[fov_radius, 0, 0], [0, fov_radius, 0], [0, 0, -fov_radius]])
        rotation = rotation_matrix(azimuth, polar)
        rotated_vectors = fov_vectors @ rotation.T
        fov_corners = position[:3] + rotated_vectors

        for corner in fov_corners:
            ax.plot([position[0], corner[0]], [position[1], corner[1]], [position[2], corner[2]], color='orange')

    def draw_camera_norm(ax, camera_positions):
        norm_c = np.array([np.cos(camera_positions[4])*np.cos(camera_positions[3]),np.sin(camera_positions[4]),np.cos(camera_positions[4])*np.sin(camera_positions[3])])
        X = camera_positions[0]
        Y = camera_positions[1]
        Z = camera_positions[2]
        U = 25000*norm_c[0]
        V = 25000*norm_c[1]
        W = 25000*norm_c[2]
        ax.quiver(X,Y,Z,U,V,W)

    for i in range(len(camera_positions)):
        ax.scatter(camera_positions[i][0], camera_positions[i][1], camera_positions[i][2], marker='.', label="Camera " + str(i + 1))
        # draw_camera_fov(ax, camera_positions[i][:3], camera_positions[i][3], camera_positions[i][4])
        draw_camera_norm(ax, camera_positions[i])

    ax.scatter(0,0,0,marker='+',linewidth=2,label='PFRR')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_xlim(-100000, 200000)
    ax.set_ylim(-200000, 100000)
    ax.set_zlim(-1000, 200000)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # # Load the image
    # from matplotlib.image import imread
    # image = imread('C:/Users/jared/OneDrive/Desktop/GPOE2025/testmap.png')

    # # Define the coordinates for the image plane
    # x = np.linspace(-100000, 200000, image.shape[1])
    # y = np.linspace(-200000, 100000, image.shape[0])
    # X, Y = np.meshgrid(x, y)
    # Z = np.zeros_like(X)-10000

    # # Plot the image as a surface
    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=image, shade=False)

    # # temp from sandbox
    # for x in range(volume.shape[0]):
    #     for y in range(volume.shape[1]):
    #         for z in range(volume.shape[2]):
    #             # Calculate the distance from the center
    #             distance = np.sqrt((x - sphere_position[0]/voxel_size)**2 + (y - sphere_position[1]/voxel_size)**2 + (z - sphere_position[2]/voxel_size)**2) #voxelsize*x ?
    #             if distance < sphere_radius/voxel_size:
    #               ax.scatter(voxel_size*x, voxel_size*y, voxel_size*z, color='green', alpha=0.1, label='Synthetic Sphere')

    plt.show()

import structuredlight as sl
import numpy as np
import cv2 as cv



def find_point_map(proj, cam):
    '''recovers the 3d world points using projected/captured
    images and known extrinsic and intrinsic parameters of
    the projector and camera.'''
    # decode indices for the images
    proj_images = []
    for img in proj.projected_imgs:
        proj_images.append(cv.cvtColor(np.array(img).astype('uint8'), cv.COLOR_BGR2GRAY))

    cam_images = []
    for img in cam.captured_imgs:
        cam_images.append(cv.cvtColor(np.array(img).astype('float32'), cv.COLOR_BGR2GRAY))

    img_indices = decode_images(proj_images, cam_images)

    # find light-plane normal vectors for each code
    n_for_index = calc_plane_norms(proj)

    # find depth using depth equation
    shape = np.shape(img_indices)
    depths = np.zeros(shape)
    point_map = np.zeros((shape[0], shape[1], 3))
    for i_r, row in enumerate(img_indices):
        for i_c, index in enumerate(row):
            # handle null values, i.e., outside the area of interest
            if index is not None:
                n = n_for_index[index]
                u = np.array([i_c, i_r, 1])
                depth = np.abs(calculate_depth(cam.T, n, cam.R, cam.Kinv, u))
                depths[i_r, i_c] = depth
                point_map[i_r, i_c] = cam.T + calculate_3d_point(cam.R, cam.Kinv, u, depth)

    return depths, point_map    


def calc_plane_norms(proj):
    '''takes the projector object and calculates the vectors normal
    to the projector light-planes of the pixel centres.'''
    # remember old information to avoid recalculating
    original_lines = proj.pix_lines
    u0 = proj.K[0][2]

    # find centre lines
    proj.width -= 1
    u0_n = (proj.width - 1) / 2
    proj.K[0][2] = u0_n
    proj.u0 = u0_n
    proj.Kinv = np.linalg.inv(proj.K)
    
    proj.calc_pix_lines() # recalculate for centres
    n_for_index = [np.cross(line.b, np.array([0, -1, 0])) for line in proj.pix_lines[0]]

    proj.width += 1 # reset to values prior to running
    proj.pix_lines = original_lines
    proj.K[0][2] = u0
    proj.u0 = u0
    proj.Kinv = np.linalg.inv(proj.K)
    return n_for_index


def calculate_depth(T, n, R, Kinv, u):
    '''takes intrinsic parameters, light plane normal, and pixel
    coordinates, and returns depth of the point.'''
    depth = - np.dot(T, n) / np.dot((R @ Kinv @ u), n)
    return depth


def calculate_3d_point(R, Kinv, u, depth):
    '''takes camera calibration matrix, pixel homogeneous coordinates,
    and calculated depth, and recovers 3d point in space.'''
    point = depth * R @ Kinv @ u
    return point


def convert_bnw_to_rgb(images):
    '''takes images in black and white format and converts
    each to the same image in rgb format.'''
    rgb_images = []
    for image in images:
        rgb_images.append([])
        for row in image:
            rgb_images[-1].append([])
            for pixel in row:
                intensity = pixel / 255
                rgb_images[-1][-1].append([intensity, intensity, intensity])
    return [np.array(image) for image in rgb_images]
    

def decode_images(projected, captured):
    '''finds the indices across the images, numbering in correct order.
    assumes grey coding in vertical strips.'''
    proj_indices = get_unique_indices(projected)
    cam_indices = get_unique_indices(captured)
    # indices may not be numbered in order.
    mapping_dict = {}
    for i, val in enumerate(proj_indices[0]): # each row is the same; vertical strips.
        mapping_dict[val] = i
    # convert grey coding format (0, 1, 3, 2, ..) to linear format. (0, 1, 2, 3, ...)
    ordered_indices = []
    for row in cam_indices:
        ordered_indices.append([])
        for index in row:
            if np.isnan(index):
                ordered_indices[-1].append(None)
            else:
                ordered_indices[-1].append(mapping_dict[index])
    return ordered_indices


def get_unique_indices(images):
    '''using grey coding, gets unique index for each pixel in a sequence
    of images. note these indices may not be in order--they are simply unique.'''
    num_imgs = len(images)
    img_indices = np.zeros(np.shape(images[0]))
    for n, image in enumerate(images):
        for i_r, row in enumerate(image):
            for i_c, pixel in enumerate(row):
                if pixel == 1:
                    img_indices[i_r, i_c] += 2**(num_imgs -1 - n)
                elif pixel != 0: # binary decoding therefore only two options.
                    img_indices[i_r, i_c] = None # anything other than 1 or 0 is None.

    return img_indices
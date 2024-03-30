import numpy as np
import matplotlib.pyplot as plt


from depth_calculation_functions_nonparallel import projector, camera, convert_indices_to_depths, data_3d
from structured_light_testing import project_and_decode
from visualiser import plot_depths, coloured_scatter
import numpy as np
import cv2 as cv


BETA = 1/2
X, Y, Z = 100, 0, 0  # camera position relative to projector
WALL_DEPTH = 2000 # depth of wall away from projector
THETA = 0
PHI = 0
FOCAL_LEN = 1 # for simplicity
IMG_SIZES = (100, 100)
OFFSETS = (10, 10)
WALL_SIZES = (80, 80)
NUM_INDICES = 80


# the program needs to:
#     generate indices and place them in a grid of undefineds
#     take the grid of indices and process them based on parameters
#     calculate the intermediate things
#     generate from this a plot in 3d of where it thinks the points are.


def generate_wall_indices(sizes, offsets, wall_sizes, num_indices):
    # takes pixel sizes (x, y) for overall image size
    # offsets (x, y) for left top corner of wall of indices
    # and wall_sizes (x, y) for width, height of wall, in pixels.
    # returns a matrix of indices.
    start_x, start_y = offsets
    end_x, end_y = start_x + wall_sizes[0], start_y + wall_sizes[1]

    index_factor = num_indices / (end_x - start_x)
    # generate indices. nan by default.
    indices = np.full(sizes, np.nan)
    for i_row, row in enumerate(indices):
        for i_col in range(len(row)):
            if i_row >= start_y and i_col >= start_x and i_row < end_y and i_col < end_x:
                indices[i_row][i_col] = index_factor * (i_col - start_x)
    return indices


# setup objects
proj = projector(BETA)
cam = camera(X, Y, Z, THETA, PHI, np.identity(3), FOCAL_LEN)

# get fake indices
indices = generate_wall_indices(IMG_SIZES, OFFSETS, WALL_SIZES, NUM_INDICES)

# calculate depths
coords = convert_indices_to_depths(indices, proj, cam)
data = data_3d([], [], [])
data.convert_coords_to_data(coords)

plot_depths(indices)
plot_depths(data.z, data.x, data.y)
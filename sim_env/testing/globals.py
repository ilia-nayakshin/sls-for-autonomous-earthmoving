import numpy as np

from test_data import proj_unique_colours
from ..general import euler_rotation_matrix


# EXTERNAL PARAMETERS
ALPHA = -0.1
BETA = 0.1
GAMMA = 0
R = euler_rotation_matrix(ALPHA, BETA, GAMMA)
T = np.array([0, 20, 20]) # translation matrix

# INTERNAL PARAMETERS
PPIX_NUM = 30 # number of projector pixels. projector is square.
CPIX_NUM = 100 # number of camera pixels. camera is square.
BETA = 0.1 # TODO: beta is a placeholder for camera/projector matrices.

# DISPLAY CONSTANTS
PLOT_PIX_LINE_RES = 10
SHOW_IMG_SIZE = (500, 500) # size images are resized to to see them clearly

# SIM ENV CONSTANTS
INDEX_LINE_RES = 1000
DATA = proj_unique_colours(PPIX_NUM, PPIX_NUM)
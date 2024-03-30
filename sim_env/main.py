import numpy as np
import cv2 as cv
import structuredlight as sl
import matplotlib.pyplot as plt

from testing.test_funcs import *
from envir_objects import bounding_box, sim_plane
from envir import sim_env
from general import euler_rotation_matrix, global_vars, project_and_capture, find_light_plane_line_intersection
from depth_recovery import convert_bnw_to_rgb, find_point_map, calc_plane_norms
from error_estimation import estimate_depth_error, show_error_img, show_error_bar_chart



# CONSTANTS
SHOW_IMG_SIZE = (500, 500) # size images are resized to to see them clearly

# rotation matrix calculation
ALPHA = -0.1
BETA = 0.1
GAMMA = 0
R = euler_rotation_matrix(ALPHA, BETA, GAMMA)
T = np.array([25, 25, -40]) # translation matrix
# TODO: beta is a placeholder for camera/projector matrices.
BETA = 0.1 # currently the same for camera and projector.
INDEX_LINE_RES = 1000
PLOT_PIX_LINE_RES = 10


# SIM ENV CONSTANTS
bbox = bounding_box([[-25, 25], [-25, 25], [0, 300]])
cambbox = bounding_box([bbox.xb, bbox.yb, [T[2], bbox.zb[1]]])

# setup simulation environment
sim = sim_env()
# surface = test_plane(200, np.array([0, 0, 1]))
# surface = test_cylinder
surface = test_corrugations
sim.add_surfaces([surface], [bbox], [30])

gv = global_vars()
gv.T = T
gray = sl.Gray()
gv.DATA = convert_bnw_to_rgb(gray.generate((gv.PPIX_NUM, gv.PPIX_NUM)))
project_and_capture(sim, gv, bbox, cambbox, True, False, False, False, False, False)
depths, point_map = find_point_map(sim.projector, sim.camera)

# show points
points = sim.flatten_point_grid_array(point_map)
sim.add_points(points)
sim.plot_points()

# get error
total, mean, errors = estimate_depth_error(point_map, surface)
print('Total Error:', total)
print('Mean Error:', mean)
# show_error_img(errors, gv.SHOW_IMG_SIZE)
# show_error_bar_chart(errors)

# show camera and translation matrix
sim.show_projcam_locations()
sim.show_camera_translation()

# show image matrix and key line
# sim.plot_image_plane(100)
# sim.plot_R_Kinv_u(np.array([3, 3, 1]), gv.PLOT_PIX_LINE_RES, 200)

# plot light plane
# sim.calc_projector_pixel_centres(gv.INDEX_LINE_RES)
# sim.show_light_plane(2)

sim.plot_pix_planes()
# sim.plot_surfaces()
sim.plot_corner_lines(cambbox, gv.PLOT_PIX_LINE_RES, bbox, gv.PLOT_PIX_LINE_RES, True)
plt.show()
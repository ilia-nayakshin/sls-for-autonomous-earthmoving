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
from saving import *



# CONSTANTS
SHOW_IMG_SIZE = (500, 500) # size images are resized to to see them clearly
INDEX_LINE_RES = 1000
PLOT_PIX_LINE_RES = 10
L_S = 20

# setup variables
ppix_num = 80 # this is 'N'
a = 2
cpix_num = round(ppix_num * (1 + a))

t = np.array([1000, 0, -1000])
zbar = 100 # surface distance
ralpha = 0 # no rotation about z axis to maximise image capture

theta_x = np.arctan(t[0] / (zbar - t[2]))
theta_y =  np.arctan(t[1] / (zbar - t[2]))
rbeta = -theta_x
rgamma = theta_y

# 'beta' ratio parameters. used to construct camera/projection matrices.
beta_p = L_S / (2 * zbar)
beta_c_x = np.tan(theta_x - np.arctan((np.abs(t[0]) - (L_S / 2)) / (zbar - t[2])))
beta_c_y = np.tan(theta_y - np.arctan((np.abs(t[1]) - (L_S / 2)) / (zbar - t[2])))

print('beta p, c_x, c_y:', beta_p, beta_c_x, beta_c_y, 'r a,b,c:',ralpha, rbeta, rgamma)

# setup projection images
gray = sl.Gray()
data = convert_bnw_to_rgb(gray.generate((ppix_num, ppix_num)))

# setup global variables
gv = global_vars(ralpha, rbeta, rgamma, None, t, ppix_num, cpix_num,
                 beta_p, beta_p, beta_c_x, beta_c_y, 1, None, None, 
                 PLOT_PIX_LINE_RES, SHOW_IMG_SIZE, INDEX_LINE_RES, data)

# SIM ENV CONSTANTS
fact_offset = 1.5
edge = (L_S / 2) * fact_offset # small offsets so no lines are ignored by accident 
bbox = bounding_box([[-edge, edge], [-edge, edge], [0, zbar * fact_offset]])
cambbox = bounding_box([[min(-edge, -t[0]), max(edge, t[0])],
                        [min(-edge, -t[1]), max(edge, t[1])], 
                        [t[2], bbox.zb[1]]])

# setup simulation environment
sim = sim_env()
surface = test_plane(zbar, np.array([0, 0, 1]))
sim.add_surfaces([surface], [bbox], [30])

project_and_capture(sim, gv, bbox, cambbox, True, False, False, False, False, False)
depths, point_map = find_point_map(sim.projector, sim.camera)

# show points
points = sim.flatten_point_grid_array(point_map)
sim.add_points(points)
sim.plot_points()

# get error
total, mean, errors, mean_diff = estimate_depth_error(point_map, surface)
print('Total Error:', total)
print('Mean Error:', mean)
print('Mean Diff:', mean_diff)
# show_error_img(errors, gv.SHOW_IMG_SIZE)
show_error_bar_chart(errors)

# show camera and translation matrix
sim.show_projcam_locations()
sim.show_camera_translation()

# show image matrix and key line
sim.plot_image_plane(25)
# sim.plot_R_Kinv_u(np.array([2, 2, 1]), gv.PLOT_PIX_LINE_RES, 100)

# # plot light plane
# sim.calc_projector_pixel_centres(gv.INDEX_LINE_RES)
# sim.show_light_plane(0)

# sim.plot_pix_lines(cambbox, PLOT_PIX_LINE_RES, True)
sim.plot_pix_planes()
# sim.projector.plot_pix_lines(sim.ax, bbox, PLOT_PIX_LINE_RES, True)
# sim.plot_surfaces()
sim.plot_corner_lines(cambbox, gv.PLOT_PIX_LINE_RES, bbox, gv.PLOT_PIX_LINE_RES, True)
plt.show()

save_error_details('n-{n}_a-{a}_t-{t_x}-{t_y}-{t_z}.npz'.format(n=ppix_num, 
                                                                a=a, t_x=t[0],
                                                                t_y=t[1], t_z=t[2]), 
                                                                sim, errors, gv)


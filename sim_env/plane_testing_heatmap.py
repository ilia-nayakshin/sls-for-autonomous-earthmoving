import numpy as np
import cv2 as cv
import structuredlight as sl
import matplotlib.pyplot as plt

# from alive_progress import alive_bar

from testing.test_funcs import *
from envir_objects import bounding_box, sim_plane
from envir import sim_env
from general import euler_rotation_matrix, global_vars, project_and_capture, find_light_plane_line_intersection
from depth_recovery import convert_bnw_to_rgb, find_point_map, calc_plane_norms
from error_estimation import estimate_depth_error, show_error_img, show_error_bar_chart
from saving import *
from estimated_error import heat_map



# CONSTANTS
SHOW_IMG_SIZE = (500, 500) # size images are resized to to see them clearly
INDEX_LINE_RES = 1000
PLOT_PIX_LINE_RES = 10
L_S = 20

# angle = 60
# SLOPE = (np.pi / 180) * angle

# setup variables
ppix_num = 20 # this is 'N'
a = 0
cpix_num = round(ppix_num * (1 + a))
zbar = 100 # surface distance
# xdiff = 4
sigma = 3.33333
zdiff = 5.496 # mound height.
ralpha = 0 # no rotation about z axis to maximise image capture

print(sigma, zdiff)

x = np.arange(-130, 140, 10)
z = np.arange(-50, 100, 10)
xx, zz = np.meshgrid(x, z)
error = np.zeros(np.shape(xx))
for i_r, row in enumerate(xx):
    print("========================== ROW ", i_r, "OF", len(xx), "==========================")
    # with alive_bar(len(row)) as row_bar:
    for i_c, x_i in enumerate(row):

        t = np.array([x_i, 0, zz[i_r, i_c]])

        theta_x = np.arctan(t[0] / (zbar - t[2]))
        theta_y =  np.arctan(t[1] / (zbar - t[2]))
        rbeta = -theta_x
        rgamma = theta_y

        # 'beta' ratio parameters. used to construct camera/projection matrices.
        beta_p = L_S / (2 * zbar)
        beta_c_x = np.tan(np.abs(theta_x) - np.arctan((np.abs(t[0]) - (L_S / 2)) / (zbar - t[2])))
        beta_c_y = np.tan(np.abs(theta_y) - np.arctan((np.abs(t[1]) - (L_S / 2)) / (zbar - t[2])))

        if theta_x < 0: beta_c_x *= -1
        if theta_y < 0: beta_c_y *= -1


        # print('beta p, c_x, c_y:', beta_p, beta_c_x, beta_c_y, 'r a,b,c:',ralpha, rbeta, rgamma)

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
        # surface = test_plane(zbar, np.array([0, 0, 1])) # simple plane test
        # surface = test_trench_x(zbar, zdiff, xdiff, SLOPE)
        surface = test_2d_gaussian(zbar, zdiff, sigma)
        # surface = test_corrugations(zbar, 4, 4, 10, 10)
        sim.add_surfaces([surface], [bbox], [30])

        sim = project_and_capture(sim, gv, bbox, cambbox, False, False, False, False, False, False)
        depths, point_map = find_point_map(sim.projector, sim.camera)

        # show points
        points = sim.flatten_point_grid_array(point_map)
        sim.add_points(points)

        # get error
        total, mean, errors = estimate_depth_error(point_map, surface)
        print('Mean Error:', mean)

        # save_error_details('emb_delz{delz}_ang{angle}_n{n}_a-{a}_t-{t_x}-{t_y}-{t_z}.npz'.format(delz=zdiff, angle=angle, n=ppix_num, 
        #                                                                 a=a, t_x=t[0],
        #                                                                 t_y=t[1], t_z=t[2]), 
        #                                                                 sim, errors, gv)
    
        error[i_r, i_c] = mean
            # row_bar() # progress indicator.

heat_map(xx, zz, error)

np.savez('error_mound_heatmap_n{n}_a{a}_scale{scale}_sigma{sigma}'.format(n=ppix_num, a=a, scale=zdiff, sigma=sigma), 
         error=error, n=ppix_num, a=a, zdiff=zdiff, sigma=sigma)


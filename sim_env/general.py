import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from testing.test_data import proj_unique_colours


class global_vars():
    '''contains global constants'''
    def __init__(self, ralpha=None, rbeta=None, rgamma=None, r=None, t=None, ppix_num=None, 
                 cpix_num=None, beta_p_x=None, beta_p_y=None, beta_c_x=None, beta_c_y=None, 
                 foc_len=None, cam_mat=None, proj_mat=None, plot_pix_line_res=None, 
                 show_img_size=None, index_line_res=None, data=None):
        # default parameters:
        # EXTERNAL PARAMETERS
        if ralpha is not None: self.rALPHA = ralpha 
        else: self.rALPHA = 0
        if rbeta is not None: self.rBETA = rbeta
        else: self.rBETA = -0.1
        if rgamma is not None: self.rGAMMA = rgamma
        else: self.rGAMMA = 0.09

        if r is not None: self.R = r
        else: self.R = euler_rotation_matrix(self.rALPHA, self.rBETA, self.rGAMMA)
        if t is not None: self.T = t
        else: self.T = np.array([0, 0.1, 0.1]) # translation matrix

        # INTERNAL PARAMETERS
        if ppix_num is not None: self.PPIX_NUM = ppix_num
        else: self.PPIX_NUM = 50 # number of projector pixels. projector is square.
        if cpix_num is not None: self.CPIX_NUM = cpix_num
        else: self.CPIX_NUM = 50 # number of camera pixels. camera is square.

        # setup for finding camera/projector matrices
        if beta_p_x is not None: self.BETA_P_X = beta_p_x
        else: self.BETA_P_X = 0.1
        if beta_p_y is not None: self.BETA_P_Y = beta_p_y
        else: self.BETA_P_Y = 0.1
        if beta_c_x is not None: self.BETA_C_X = beta_c_x
        else: self.BETA_C_X = 0.1
        if beta_c_y is not None: self.BETA_C_Y = beta_c_y
        else: self.BETA_C_Y = 0.1
        if foc_len is not None: self.FOC_LEN = foc_len
        else: self.FOC_LEN = 1

        # CAMERA AND PROJECTOR MATRICES
        # projector pixel lines are calculated at the corners so need no adjustment
        proj_u0, proj_v0 = self.PPIX_NUM / 2, self.PPIX_NUM / 2
        if proj_mat is not None: self.PROJ_MAT = proj_mat
        else: self.PROJ_MAT = get_camera_matrix_from_beta(self.BETA_P_X, self.BETA_P_Y, self.FOC_LEN, 
                                                          self.FOC_LEN, self.PPIX_NUM, self.PPIX_NUM, 
                                                          proj_u0, proj_v0)
        # camera pixel lines lie in the centre so are adjusted accordingly
        cam_u0, cam_v0 = (self.CPIX_NUM - 1) / 2, (self.CPIX_NUM - 1) / 2
        if cam_mat is not None: self.CAM_MAT = cam_mat
        else: self.CAM_MAT = get_camera_matrix_from_beta(self.BETA_C_X, self.BETA_C_Y, self.FOC_LEN,
                                                          self.FOC_LEN, self.CPIX_NUM, self.CPIX_NUM, 
                                                          cam_u0, cam_v0)

        # DISPLAY CONSTANTS
        if plot_pix_line_res is not None: self.PLOT_PIX_LINE_RES = plot_pix_line_res
        else: self.PLOT_PIX_LINE_RES = 10
        if show_img_size is not None: self.SHOW_IMG_SIZE = show_img_size
        else: self.SHOW_IMG_SIZE = (500, 500) # size images are resized to to see them clearly

        # SIM ENV CONSTANTS
        if index_line_res is not None: self.INDEX_LINE_RES = index_line_res
        else: self.INDEX_LINE_RES = 1000
        if data is not None: self.DATA = data
        else: self.DATA = proj_unique_colours(self.PPIX_NUM, self.PPIX_NUM)

        


def setup_3d():
    '''generates a 3d figure and returns the axis.'''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.set_xlabel("x direction")
    ax.set_ylabel("y direction")
    ax.set_zlabel("z direction")
    return ax



def euler_rotation_matrix(alpha, beta, gamma):
    r_x = np.array([[1, 0, 0], [0, np.cos(gamma), -np.sin(gamma)], [0, np.sin(gamma), np.cos(gamma)]])
    r_y = np.array([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
    r_z = np.array([[np.cos(alpha), -np.sin(alpha), 0], [np.sin(alpha), np.cos(alpha), 0], [0, 0, 1]])
    return np.matmul(r_z, np.matmul(r_y, r_x))



def get_camera_matrix_from_beta(beta_x, beta_y, foc_len_x, foc_len_y, pix_num_x, pix_num_y, u0, v0):
    '''returns camera matrix K for projection ratio (focus / width) beta.'''
    pix_len_x = beta_x / (pix_num_x / 2)
    pix_len_y = beta_y / (pix_num_y / 2)
    K = np.array([[foc_len_x / pix_len_x, 0, u0],
                  [0, foc_len_y / pix_len_y, v0],
                  [0, 0, 1]])
    return K


def check_line_within_bounds(line, bounds=None):
    # calculate possible lambdas for edges of bounds
    if bounds == None:
        bb = line.bounds
    else:
        bb = bounds
    a, b = line.a, line.b
    x_min = np.array([bb.xb[0], bb.yb[0], bb.zb[0]])
    x_max = np.array([bb.xb[1], bb.yb[1], bb.zb[1]])
    lambda_mins = np.divide(x_min - a, b)
    lambda_maxs = np.divide(x_max - a, b)
    lambda_min, lambda_max = np.nan, np.nan
    # iterate over all possible lambdas
    for lamb in np.concatenate((lambda_mins, lambda_maxs), axis=None):
        if not np.isinf(lamb):
            if np.isnan(lamb):
                lamb = 0 # correct any divide by 0 errors
            # verify lambda results in within-bounds result
            if (a + lamb * b >= x_min).all() and (a + lamb * b <= x_max).all():
                if np.isnan(lambda_min):
                    lambda_min, lambda_max = lamb, lamb
                if lamb < lambda_min:
                    lambda_min = lamb
                elif lamb > lambda_max:
                    lambda_max = lamb
    # check both maximum and minimum have been found
    if lambda_min == lambda_max:
        print("Error: line out of bounds.")
        return False
    else:
        return lambda_min, lambda_max


def find_light_plane_line_intersection(n, line):
    '''takes a light plane with equation x.n = 0 and sim_line
    object and finds their intersection.'''
    if np.dot(line.b, n) != 0 :
        lamb = - line.a.dot(n) / line.b.dot(n)
        return line.a + lamb * line.b
    else:
        print('Error: line and plane are parallel')
        return False


def plot_points(ax, points, display_text=None):
    '''simple function to scatter plot points in [[x, y, z], [x, y, z], ...] format.
    if display_text is provided, also provides labels to each point.'''
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    z = [point[2] for point in points]
    ax.scatter(x, y, z)
    if display_text is not None:
        for i_p in range(len(points)):
            ax.text(x[i_p], y[i_p], z[i_p], display_text[i_p])


def plot_line(ax, line, colour=False, dashed=False):
    '''plots a line with equation x = a + lambda b, within boundaries [low, high]'''
    a, b, resolution = line.a, line.b, line.resolution
    within_bounds = check_line_within_bounds(line)

    # handling styles
    if dashed:
        dashed = 'dashed'
    else:
        dashed = 'solid'

    if within_bounds != False:
        lambda_min, lambda_max = within_bounds # unpack lambdas
        lambdas = np.linspace(lambda_min, lambda_max, resolution)
        xs = np.array([a + lamb * b for lamb in lambdas])
        if colour is None:
            ax.plot(xs[:, 0], xs[:, 1], xs[:, 2], color='black', linestyle=dashed)
        elif colour:
            ax.plot(xs[:, 0], xs[:, 1], xs[:, 2], color=line.info, linestyle=dashed)
        else:
            ax.plot(xs[:, 0], xs[:, 1], xs[:, 2], linestyle=dashed)


def calculate_nearest_point(a, b, lambs, surf_func):
    '''function to find the intersection of a line and an arbitrary surface.
    imperfect as it does not use normals to the surface, but generally sufficient
    for this application.'''
    min_error = np.inf # set initially to infinite error.
    for lamb in lambs:
        point = a + lamb * b # point x on the line
        surf_point = np.array(surf_func(point[0], point[1], point[2]))
        # convert from meshgrid formatting to point
        if np.shape(surf_point) != np.shape(np.zeros(3)):
            surf_point = np.array([surf_point[0][0][0], surf_point[1][0][0], surf_point[2][0][0]])
        error = np.linalg.norm(point - surf_point)
        if error < min_error:
            min_error = error
            nearest_point = point
    return nearest_point


### very nice function but less useful now because of plot_surface.
def plot_plane(ax, plane):
    '''plots a plane with equation x.n = d'''
    if plane.check_plane_within_bounds():
        # find corners of the subplane being plotted.
        points = plane.find_bb_intersections()
        points = plane.bounds.remove_external_points(points)
        u, v = plane.find_normal_plane_vectors()
        lambdas = plane.find_lambdas_for_points(points)
        lamb_1_min, lamb_2_min = lambdas[0][0], lambdas[0][0]
        lamb_1_max, lamb_2_max = lambdas[0][1], lambdas[0][1]
        for lamb in lambdas:
            if lamb[0] < lamb_1_min:
                lamb_1_min = lamb[0]
            elif lamb[0] > lamb_1_max:
                lamb_1_max = lamb[0]
            if lamb[1] < lamb_2_min:
                lamb_2_min = lamb[1]
            elif lamb[1] > lamb_2_max:
                lamb_2_max = lamb[1]
        # create grid of points on sub-plane.
        lamb_1s = np.linspace(lamb_1_min, lamb_1_max, plane.resolution)
        lamb_2s = np.linspace(lamb_2_min, lamb_2_max, plane.resolution)
        point_grid = []
        for lamb_1 in lamb_1s:
            for lamb_2 in lamb_2s:
                point_grid.append(plane.d * plane.n + lamb_1 * u + lamb_2 * v)
        point_grid = plane.bounds.remove_external_points(point_grid)
        # plot grid.
        point_grid = np.array(point_grid)
        x, y, z = point_grid[:, 0], point_grid[:, 1], point_grid[:, 2]
        ax.scatter(x, y, z)
    else:
        print("Error: plane out of bounds.")


def plot_surface(ax, func, bounds, resolution, display='surface', opacity=0.5):
    '''plots a 3d scatter plot within the bounding box of
    func(x, y, z). func must return x, y, z. note that if
    the function exceeds the bounding box, the surface will
    plot this anyway. e.g. z = x+y between -5 and 5, z can
    exceed -5, 5, if not taken into account by func.'''
    x = np.linspace(bounds.xb[0], bounds.xb[1], resolution)
    y = np.linspace(bounds.yb[0], bounds.yb[1], resolution)
    z = np.linspace(bounds.zb[0], bounds.zb[1], resolution)
    x, y, z = func(x,y,z)
    if display == 'surface':
        ax.plot_surface(x, y, z, alpha=opacity)
    elif display == 'scatter':
        ax.scatter(x, y, z)
    else:
        print("Error: invalid display option.")


def project_and_capture(sim, gvs, bbox, cambbox, show_images=False, show_pix_lin=False, 
                        show_corn_lin=False, show_pix_plan=False, show_planes=False, show_surfs=False):
    '''Main projecting and capturing loop of the code. Configurable options for what to show.
    If none of these options are true, the program will not display anything. Note that the
    global variables used in this function must be defined in order for this code to run.'''
    # setup camera and projector
    sim.setup_projector(gvs.PROJ_MAT, gvs.PPIX_NUM, gvs.PPIX_NUM)
    sim.setup_camera(gvs.CAM_MAT, gvs.FOC_LEN, gvs.CPIX_NUM, gvs.CPIX_NUM, gvs.R, gvs.T)

    # calculate indices for grid
    sim.calc_proj_pix_indices_for_camera_grid(gvs.INDEX_LINE_RES, False)
    
    # setup data - this may be only one image or it may be multiple.
    if len(np.shape(gvs.DATA)) == 3:
        gvs.DATA = [gvs.DATA] # wrap for compatibility with next loop
    for img in gvs.DATA:
        sim.set_proj_info(img)
        sim.get_cam_info()

        # display images projected and captured.
        if show_images:
            try:
                sim.show_info_img('proj', new_size=gvs.SHOW_IMG_SIZE)
                sim.show_info_img('cam', new_size=gvs.SHOW_IMG_SIZE)
            except cv.error:
                print(cv.error)

        # display results
        if show_pix_lin: sim.plot_pix_lines(cambbox, gvs.PLOT_PIX_LINE_RES, True)
        if show_corn_lin: sim.plot_corner_lines(cambbox, gvs.PLOT_PIX_LINE_RES, bbox, gvs.PLOT_PIX_LINE_RES, True)
        if show_pix_plan: sim.plot_pix_planes()
        if show_planes: sim.plot_planes()
        if show_surfs: sim.plot_surfaces()
        if show_pix_lin or show_corn_lin or show_pix_plan or show_planes or show_surfs: plt.show()
        else: # hold images only if not plotting.
            if show_images: cv.waitKey(0)

    # return updated simulation environment.
    return sim


# function taken from online for testing.
def gradient_descent(surf_func, line, delta_lamb, lamb_min, lamb_max, 
                     n_iter, learn_rate=0.1, tol=1e-08):
    '''function to perform gradient descent to find the intersection
    between a surface function and a line.'''

    lamb = (lamb_min + lamb_max) / 2
    min_error = np.inf
    for i in range(n_iter):
        # find main point and points above & below for gradient.
        point = line.a + lamb * line.b
        # point = x.T
        point_abv = np.array(point + delta_lamb * line.b)
        point_bel = np.array(point - delta_lamb * line.b)

        # find corresponding surface points
        surf_point = surf_func(point[0], point[1], point[2])
        surf_point_abv = surf_func(point_abv[0], point_abv[1], point_abv[2])
        surf_point_bel = surf_func(point_bel[0], point_bel[1], point_bel[2])
        # reformat from meshgrid arrays
        surf_point = [surf_point[0][0][0], surf_point[1][0][0], surf_point[2][0][0]]
        surf_point_abv = [surf_point_abv[0][0][0], surf_point_abv[1][0][0], surf_point_abv[2][0][0]]
        surf_point_bel = [surf_point_bel[0][0][0], surf_point_bel[1][0][0], surf_point_bel[2][0][0]]

        # find corresponding errors
        error = np.linalg.norm(surf_point - point)**2
        error_abv = np.linalg.norm(surf_point_abv - point_abv)**2
        error_bel = np.linalg.norm(surf_point_bel - point_bel)**2

        # find gradient
        grad = (error_abv - error_bel) / (2*delta_lamb)
        # find new lambda
        diff = - learn_rate * grad
        lamb += diff

        # update error and point if necessary
        if error < min_error:
            min_error = error
            nearest_point = point
        
        # end loop if necessary
        if error <= tol or np.abs(diff) <= tol or lamb > lamb_max or lamb < lamb_min:
            # print((diff <= tol), (lamb > lamb_max), (lamb < lamb_min), nearest_point)
            break
    if np.abs(min_error) > tol and np.abs(diff) > tol: 
        # pass
        print(i, min_error, "Error above threshold. Try increasing threshold or changing learning rate.")
    return nearest_point
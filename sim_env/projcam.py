import numpy as np

from alive_progress import alive_bar

from envir_objects import bounding_box, sim_line, sim_plane
from general import check_line_within_bounds, calculate_nearest_point, plot_line, gradient_descent, get_camera_matrix_from_beta


# constants for learning
LEARN_RATE = 0.01


class sim_projcam():
    '''parent class for cameras and projectors. contains some
    common methods for the two.'''

    def calc_pix_lines(self):
        '''calculates the unit vectors of all pixel "lines"'''

        self.pix_lines = []
        # check if odd or even and find maximum deviation from camera centre
        for row in range(self.height):
            n_row = row - self.v0
            self.pix_lines.append([])
            for col in range(self.width):
                n_col = col - self.u0
                b = np.array([n_col * self.Kinv[0][0],
                                n_row * self.Kinv[1][1],
                                1])
                line = sim_line(self.T, self.R.dot(b), None, None)
                self.pix_lines[-1].append(line)

    def calc_intersections(self, simulation_env, resolution):
        '''computes all intersections (or approximate intersections) between
        pixel lines and the planes/surfaces given.'''
        plane_intersections = []
        surf_intersections = []

        # error handling.
        if len(simulation_env.planes) == 0:
            if len(simulation_env.surfaces) == 0:
                print('Error: No planes or surfaces set.')
            else:
                print('Warning: No planes set.')
        elif len(simulation_env.surfaces) == 0:
            print('Warning: No surfaces set.')
        
        # iterate over all planes
        for plane in simulation_env.planes:
            plane_intersections.append([])
            for row in self.pix_lines:
                # maintain tabular structure within intersection points
                # to maintain pixel indices
                plane_intersections[-1].append([])
                for line in row:
                    lamb_p_i = (plane.d - self.T.dot(plane.n)) / (line.b.dot(plane.n))
                    plane_intersections[-1][-1].append(self.T + lamb_p_i * line.b)
        
        # iterate over all surfaces
        for i, surf in enumerate(simulation_env.surfaces):
            surf_intersections.append([])
            with alive_bar(len(self.pix_lines)) as bar:
                for row in self.pix_lines:
                    surf_intersections[-1].append([])
                    for line in row:
                        lamb_min, lamb_max = 0, simulation_env.surf_bounds[i].find_max_dist()
                        d_lamb = (lamb_max - lamb_min) / resolution
                        point = gradient_descent(surf, line, d_lamb, lamb_min, lamb_max, resolution, LEARN_RATE)
                        surf_intersections[-1][-1].append(point)
                    bar()
        
        # combine two lists
        intersections = plane_intersections + surf_intersections
        return intersections
    
    def plot_corner_lines(self, ax, bounds, res, with_colours, dashed):
        corner_lines = [self.pix_lines[0][0],
                        self.pix_lines[0][-1],
                        self.pix_lines[-1][0],
                        self.pix_lines[-1][-1]]
        for line in corner_lines:
            line.bounds = bounds
            line.resolution = res
            plot_line(ax, line, with_colours, dashed)



class sim_projector(sim_projcam):
    '''contains additional functions specific to projectors, 
    such as calculating within which pixel a point sits. a
    'simulation' projector. the projector has centre at 
    0,0,0 and faces towards the positive z-direction.'''

    def __init__(self, K, n_pix_width, n_pix_height):
        self.K = K
        self.u0 = K[0][2]
        self.v0 = K[1][2]
        self.Kinv = np.linalg.inv(self.K)
        self.width = n_pix_width + 1 # pixel corners, not centres
        self.height = n_pix_height + 1 # as above
        self.T = np.zeros(3)
        self.R = np.eye(3)
        self.projected_imgs = []

    def assign_pix_info(self, pix_info):
        '''takes 2D array of pixel info and assigns to self.pix_info.'''
        self.pix_info = pix_info
        self.projected_imgs.append(pix_info)
    
    def get_pix_info(self):
        '''fetches information stored in self.pix_info and returns as a 2D array.'''
        return np.array(self.pix_info)

    def get_plane_params(self, corners, least_squares):
        '''takes 4 corners of a plane and finds the least squares plane
        parameters. returns a sim_plane object.'''
        if least_squares:
            X = np.array(corners)
            # use least squares solution for n
            n, _, _, _ = np.linalg.lstsq(X, np.ones(len(corners)), rcond=None)
            # n = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(np.ones(len(corners)))
            n = n / np.linalg.norm(n) # normalise
            x_avg = sum(corners) / 4
            d = n.dot(x_avg)
            bbox = bounding_box([[np.min(X[:, 0]), np.max(X[:, 0])],
                                [np.min(X[:, 1]), np.max(X[:, 1])],
                                [np.min(X[:, 2]), np.max(X[:, 2])]])
            plane = sim_plane(n, d, bbox, 3)
            plane.set_corner_params(corners)
        else:
            plane_abc = self.get_triangle_subpix_plane(corners[:-1])
            plane_bcd = self.get_triangle_subpix_plane(corners[1:][::-1])
            plane_abc.set_corner_params(corners[:-1])
            plane_bcd.set_corner_params(corners[1:][::-1])
            plane = [plane_abc, plane_bcd]
        return plane
    
    def get_triangle_subpix_plane(self, corners):
        '''takes three points and finds the plane connecting them.'''
        corners = np.array(corners)
        l_ba = corners[0] - corners[1]
        l_ca = corners[0] - corners[2]
        l_ba = l_ba / np.linalg.norm(l_ba)
        l_ca = l_ca / np.linalg.norm(l_ca)
        n = np.cross(l_ba, l_ca)
        n = n / np.linalg.norm(n)
        d = np.dot(n, corners[0])
        bbox = bounding_box([[np.min(corners[:, 0]), np.max(corners[:, 0])],
                             [np.min(corners[:, 1]), np.max(corners[:, 1])],
                             [np.min(corners[:, 2]), np.max(corners[:, 2])]])
        return sim_plane(n, d, bbox, None) # resolution is not used for subpix planes.

    def get_four_corners(self, all_corners):
        '''converts n x m array of corners into n-1 x m-1 x 4 array
        of 4 corners per pixel.'''
        all_fourcorners = []
        for plane_corners in all_corners:
            all_fourcorners.append([])
            for i_r in range(0, len(plane_corners) - 1):
                all_fourcorners[-1].append([])
                for i_c in range(0, len(plane_corners[0]) - 1):
                    all_fourcorners[-1][-1].append([plane_corners[i_r][i_c],
                                                    plane_corners[i_r][i_c + 1],
                                                    plane_corners[i_r + 1][i_c],
                                                    plane_corners[i_r + 1][i_c + 1]])
        return all_fourcorners

    def get_all_pix_planes(self, all_fourcorners, least_squares=False):
        '''takes the plane/surface intersection corners for the projector
        and calculates all the pixel 'sub-plane' parameters.'''
        planes = []
        for simplane in all_fourcorners:
            planes.append([])
            for row in simplane:
                planes[-1].append([])
                for pixel in row:
                    planes[-1][-1].append(self.get_plane_params(pixel, least_squares))
        # reorder array to [n x m x simplanes] rather than [simplane x n x m]
        # use two triangular sub-planes if not least squares.
        if least_squares: new_planes = np.empty((len(planes[0]), len(planes[0][0]), len(planes)), dtype=sim_plane)
        else: new_planes = np.empty((len(planes[0]), len(planes[0][0]), len(planes), 2), dtype=sim_plane)
        for i_s, simplane in enumerate(planes):
            for i_r, row in enumerate(simplane):
                for i_p, pparams in enumerate(row):
                    new_planes[i_r][i_p][i_s] = pparams
        return new_planes
    
    def calc_pix_centre_lines(self, sim_env, resolution):
        '''finds the centre lines for each projector pixel, rather
        than the corner lines. sets this to self.pix_centre_lines.
        then calculates intersections with these new lines, returning them.'''
        
        # remember old intersections to avoid recalculating
        original_lines = self.pix_lines
        u0, v0 = self.K[0][2], self.K[1][2]

        # find centre lines
        self.width -= 1
        self.height -= 1 # reduce to calculate centres

        u0_n, v0_n = (self.width - 1) / 2, (self.height - 1) / 2
        self.K[0][2], self.K[1][2] = u0_n, v0_n
        self.Kinv = np.linalg.inv(self.K)
        self.u0, self.v0 = u0_n, v0_n
        self.calc_pix_lines()
        # self.plot_pix_lines(sim_env.ax, bbox, resolution, True)

        # find intersections with centre lines
        centre_intersections = self.calc_intersections(sim_env, resolution)
        

        # assign new variable and reset old ones
        self.pix_centre_lines = self.pix_lines
        self.pix_lines = original_lines
        self.width += 1
        self.height += 1
        self.K[0][2], self.K[1][2] = u0, v0
        self.u0, self.v0 = u0, v0
        self.Kinv = np.linalg.inv(self.K)
        
        return centre_intersections
    
    def plot_pix_lines(self, ax, bounds, resolution, with_colours=False, dashed=False):
        for row in self.pix_lines:
            for line in row:
                line.bounds = bounds
                line.resolution = resolution
                plot_line(ax, line, with_colours, dashed)



class sim_camera(sim_projcam):
    '''simulation camera. stores camera params and
    calculates where surfaces and pixel lines meet.'''

    def __init__(self, K, foc_len, n_pix_width, n_pix_height, R, T):
        self.width = n_pix_width
        self.height = n_pix_height
        self.foc_len = foc_len
        self.pix_len_x = foc_len / K[0][0]
        self.pix_len_y = foc_len / K[1][1]
        self.R = R
        self.T = T
        self.u0 = K[0][2]
        self.v0 = K[1][2]
        self.K = K
        self.Kinv = np.linalg.inv(self.K)
        self.captured_imgs = []

    def assign_pix_info(self, pix_info):
        '''takes 2D array of pixel info and assigns each element to
        the according pixel line in self.pix_lines.'''
        for i_r, row in enumerate(self.pix_lines):
            for i_c, line in enumerate(row):
                line.assign_info(pix_info[i_r][i_c])
        
    def get_pix_info(self):
        '''fetches all information stored in self.pix_lines and returns as a 2D array.'''
        info = []
        for row in self.pix_lines:
            info.append([])
            for item in row:
                info[-1].append(item.info)
        return np.array(info)
    
    def plot_pix_lines(self, ax, bounds, resolution, with_colours=False, dashed=False):
        for row in self.pix_lines:
            for line in row:
                line.bounds = bounds
                line.resolution = resolution
                plot_line(ax, line, with_colours, dashed)
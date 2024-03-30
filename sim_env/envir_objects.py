import numpy as np



class sim_line():
    '''contains plane with equation a + lambda b = x.
    optionally can also contain information, e.g. colour.'''

    def __init__(self, a, b, bounds, resolution, info=None):
        self.a = a
        self.b = b / np.linalg.norm(b) # normalise b
        self.bounds = bounds # bounding_box object.
        self.resolution = resolution
        self.info = info
    
    def assign_info(self, info):
        self.info = info



class sim_plane():
    '''contains information about a simulation plane and
    various other information including points, bounds, resolution,
    and normal plane vectors'''

    def __init__(self, n, d, bounds, resolution):
        self.n = n / np.linalg.norm(n) # normalise
        self.d = d
        self.bounds = bounds # bounding_box object.
        self.resolution = resolution
        
    def find_normal_plane_vectors(self):
        '''find unit normal vectors u, v on the plane with normal n.'''
        self.u = np.cross(self.n, np.array([1, 0, 0]))
        if np.linalg.norm(self.u) == 0: # handle case where n==[1, 0, 0]
            self.u = np.cross(self.n, np.array([0, 1, 0]))
        self.u = self.u / np.linalg.norm(self.u) # normalise.
        self.v = np.cross(self.n, self.u) # should have magnitude 1 by default.
        return self.u, self.v
    
    def set_corner_params(self, corners):
        '''for a plane with corners ABCD..., finds unit lines from A to B,
        B to C, C to D, ...'''
        len_c = len(corners)
        self.pcorner_lines = []
        for i_c, corner in enumerate(corners):
            line = corners[(i_c + 1) % len_c] - corner
            line = line / np.linalg.norm(line)
            self.pcorner_lines.append(line)
        self.corners = corners
    
    def find_lambdas_for_points(self, points):
        # TODO: this currently breaks for certain n since this only
        # considers two of the possible three equations.
        '''solves for the lambda values of each point such that
        d*n + lambda_1 * u + lambda_2 * v = x'''
        lambdas = []
        # solve A lambda = b for lambda coeffs of each point
        A = np.array([[self.u[0], self.v[0]],
                    [self.u[1], self.v[1]]])
        for point in points:
            b = np.array([point[0] - self.d*self.n[0], point[1] - self.d*self.n[1]])
            lambdas.append(np.linalg.solve(A, b))
        return lambdas
    
    def check_plane_within_bounds(self):
        '''checks the plane appears within the bounding box.
        returns true if it does and false if not.'''
        # calculates all 8 corners of the bounds
        corners = self.bounds.calculate_corners()
        min_found, max_found = False, False
        for corner in corners:
            if np.dot(corner, self.n) <= self.d:
                min_found = True
            else:
                max_found = True
        # check plane is within bounds.
        if min_found and max_found:
            return True # within bounds
        else:
            return False # plane does not appear inside bounding box.
        
    def find_triplane_intersection(self, x, y, z):
        '''finds the intersection of self with planes where two of x, y, z are 
        known and the remaining is None. e.g. [1, 2, 3].x = 10, y=2, z=5'''
        if x is None:
            x = (self.d - self.n[1]*y - self.n[2]*z) / self.n[0]
        elif y is None:
            y = (self.d - self.n[0]*x - self.n[2]*z) / self.n[1]
        elif z is None:
            z = (self.d - self.n[0]*x - self.n[1]*y) / self.n[2]
        else:
            print("Error: at least one of x, y, z must be None.")
            return None
        return np.array([x, y, z])
    
    def find_bb_intersections(self):
        '''finds the twelve intersection points of the plane n.x = d
        with the bounding box.'''
        x_parallel = (np.linalg.norm(np.cross(self.n, np.array([1, 0, 0]))) == 0)
        y_parallel = (np.linalg.norm(np.cross(self.n, np.array([0, 1, 0]))) == 0)
        z_parallel = (np.linalg.norm(np.cross(self.n, np.array([0, 0, 1]))) == 0)
        points = []
        if not y_parallel and not z_parallel:
            points.append(self.find_triplane_intersection(None, self.bounds.yb[0], self.bounds.zb[0]))
            points.append(self.find_triplane_intersection(None, self.bounds.yb[0], self.bounds.zb[1]))
            points.append(self.find_triplane_intersection(None, self.bounds.yb[1], self.bounds.zb[0]))
            points.append(self.find_triplane_intersection(None, self.bounds.yb[1], self.bounds.zb[1]))
        if not x_parallel and not z_parallel:
            points.append(self.find_triplane_intersection(self.bounds.xb[0], None, self.bounds.zb[0]))
            points.append(self.find_triplane_intersection(self.bounds.xb[0], None, self.bounds.zb[1]))
            points.append(self.find_triplane_intersection(self.bounds.xb[1], None, self.bounds.zb[0]))
            points.append(self.find_triplane_intersection(self.bounds.xb[1], None, self.bounds.zb[1]))
        if not x_parallel and not y_parallel:
            points.append(self.find_triplane_intersection(self.bounds.xb[0], self.bounds.yb[0], None))
            points.append(self.find_triplane_intersection(self.bounds.xb[0], self.bounds.yb[1], None))
            points.append(self.find_triplane_intersection(self.bounds.xb[1], self.bounds.yb[0], None))
            points.append(self.find_triplane_intersection(self.bounds.xb[1], self.bounds.yb[1], None))
        return points



class bounding_box():
    '''contains information about a cuboid defined by 6 planes.'''

    def __init__(self, bounds):
        self.xb, self.yb, self.zb = bounds

    def calculate_corners(self):
        corners = np.array([[self.xb[0], self.yb[0], self.zb[0]],
                            [self.xb[0], self.yb[0], self.zb[1]],
                            [self.xb[0], self.yb[1], self.zb[0]],
                            [self.xb[0], self.yb[1], self.zb[1]],
                            [self.xb[1], self.yb[0], self.zb[0]],
                            [self.xb[1], self.yb[0], self.zb[1]],
                            [self.xb[1], self.yb[1], self.zb[0]],
                            [self.xb[1], self.yb[1], self.zb[1]]])
        return corners

    def remove_external_points(self, points):
        '''takes list of points and removes all points that are outside the bounding box.'''
        final_points = []
        for point in points:
            if point[0] >= self.xb[0] and point[0] <= self.xb[1] \
                and point[1] >= self.yb[0] and point[1] <= self.yb[1] \
                and point[2] >= self.zb[0] and point[2] <= self.zb[1]:
                final_points.append(point)
        return final_points
    
    def find_max_dist(self):
        x_dist = (self.xb[1] - self.xb[0]) ** 2
        y_dist = (self.yb[1] - self.yb[0]) ** 2
        z_dist = (self.zb[1] - self.zb[0]) ** 2
        return (x_dist + y_dist + z_dist) ** 0.5
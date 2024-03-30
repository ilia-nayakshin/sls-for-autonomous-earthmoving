import numpy as np


# CONSTANTS
X, Y, Z = 5, 5, 5 # m
THETA = 0 # radians, angle around z axis
PHI = 0 # radians, angle around y axis
BETA = np.arctan(1/2) # projector angle (img width/projection distance)
FOCAL_LEN = 4.44 # mm
PIX_WIDTH, PIX_HEIGHT = 960, 540
PIX_LEN = 1.4 * 10**(-3) * (3840 / PIX_WIDTH) # scale up if running at lower resolution
# MAX_DEPTH = 2500 # capping value of depth, mm.
MAX_DEPTH = 3000



class plane():
    # holds plane parameters.
    def __init__(self, a, b, c, d=0):
        self.a = a
        self.b = b
        self.c = c
        self.d = d # passes through origin by default
        self.n = np.matrix([a, b, c])


class projector():
    # holds position and orientation of projector.
    def __init__(self, beta):
        self.beta = beta
        self.delta_beta = 2*self.beta / PIX_WIDTH
    def calculate_all_plane_params(self):
        # calculates equations for planes rotated by a certain amount about the origin
        self.planes = []
        for n in range(-PIX_WIDTH//2, PIX_WIDTH//2 + 1):
            self.planes.append(plane(np.cos(n*self.delta_beta), 0, np.sin(n*self.delta_beta)))
        return self.planes


class camera():
    # holds various camera parameters.
    def __init__(self, x, y, z, theta, phi, K, focal_len):
        self.x = x
        self.y = y
        self.z = z
        self.T = np.matrix([self.x, self.y, self.z]).T
        self.theta = theta
        self.phi = phi
        self.K = K # camera matrix containing focal lengths and optic centre
        self.focal_len = focal_len
        self.optic_centre = np.array([self.K[0, 2], self.K[1, 2]])
        self.precalc_matrices()
    def calculate_rotation_matrix(self):
        # calculates rotation matrix from projector to camera.
        self.R = np.matrix([[np.cos(self.phi),  np.sin(self.theta) * np.sin(self.phi), np.cos(self.theta) * np.sin(self.phi)], 
                            [0,                 np.cos(self.theta),                   -np.sin(self.theta)], 
                            [-np.sin(self.phi), np.sin(self.theta) * np.cos(self.phi), np.cos(self.theta) * np.cos(self.phi)]])
    def calculate_RT_T(self):
        # calculates (R transpose) T.
        self.RT_T = np.matmul(self.R.T, self.T)
    def calculate_RT_Kinv(self):
        # calculates (R transpose) K^-1. used to calculate lambda later.
        self.RT_Kinv = np.matmul(self.R.T, np.linalg.inv(self.K))
    def calculate_RT_I_T(self):
        # calculates ((R transpose) + I)T. used to calculate lambda later.
        self.RT_I_T = np.matmul(self.R.T + np.identity(3), self.T)
    def precalc_matrices(self):
        self.calculate_rotation_matrix()
        self.calculate_RT_T()
        self.calculate_RT_Kinv()
        self.calculate_RT_I_T()


class data_3d():
    def __init__(self, x, y, data):
        self.x = x
        self.y = y
        self.z = data
    def convert_coords_to_data(self, coords):
        # warning: this wipes all data from the class.
        self.x, self.y, self.z = [], [], []
        for row in coords:
            self.x.append([])
            self.y.append([])
            self.z.append([])
            for pixel in row:
                self.x[-1].append(pixel[0].item())
                self.y[-1].append(pixel[1].item())
                self.z[-1].append(pixel[2].item())
    def save_data(self, filename):
        np.save(filename, self, allow_pickle=True)


def calculate_lambda(n, camera, img_coords):
    # calculates lambda = (n . (R^-1 + I)T)/(n . (R^-1 K^-1 u)).
    # used to calculate depth. note R^-1 = R^T (rotation matrix).
    numerator = np.dot(n, camera.RT_I_T)
    denominator = np.dot(n, np.matmul(camera.RT_Kinv, img_coords))
    return numerator / denominator


def calculate_world_coordinates(n, camera, img_coords):
    if n is None:
        world_coords = np.array([img_coords[0].item(), img_coords[1].item(), np.nan])
    else:
        numerator = n.dot(camera.RT_T)
        # numerator = - camera.focal_len * (n.dot(camera.T))
        prod = np.matmul(camera.RT_Kinv, img_coords)
        denominator = prod.T.dot(n.T)
        depth = (numerator / denominator).item()
        if np.abs(depth) > MAX_DEPTH:
            depth = MAX_DEPTH * (depth / np.abs(depth))
        world_coords = np.array([img_coords[0].item(),
                                 img_coords[1].item(),
                                 depth])
    return world_coords


def convert_indices_to_depths(indices, projector, camera):
    planes = projector.calculate_all_plane_params()
    # for 600 pixels wide, we can have 512 or 1024 indices with binary. code generates 1024,
    # but we can only use 600. so scale all indices accordingly.
    num_indices = 2 ** (np.ceil(np.log2(PIX_WIDTH)))
    scale = PIX_WIDTH / num_indices
    indices = np.round(indices * scale).astype(int) # round to nearest integer
    world_coords = []
    for i_row, row in enumerate(indices):
        world_coords.append([])
        for i_col, pix_index in enumerate(row):
            if pix_index == 0:
                nth_norm = None  # for case where no light detected.
            elif pix_index >= PIX_WIDTH:
                nth_norm = planes[PIX_WIDTH - 1].n
            else:
                nth_norm = planes[pix_index].n
            coords = calculate_world_coordinates(nth_norm, camera, np.matrix([i_row, i_col, 1]).T)
            world_coords[-1].append(coords)
    return world_coords

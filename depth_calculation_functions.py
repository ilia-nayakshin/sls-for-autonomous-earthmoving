import numpy as np


# CONSTANTS
X, Y, Z = 5, 5, 5 # m
THETA = 0 # radians, angle around z axis
PHI = 0 # radians, angle around y axis
BETA = np.arctan(1/2) # projector angle (img width/projection distance)
FOCAL_LEN = 4.44 # mm
X_C, Y_C = 600, 600 # optic centre, in px. will be found by calibration.
PIX_WIDTH, PIX_HEIGHT = 960, 540
# MAX_DEPTH = 2500 # capping value of depth, mm.
MAX_DEPTH = 999999



class plane():
    # holds plane parameters.
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.params = np.array([a, b, c, d])


class projector():
    # holds position and orientation of projector.
    def __init__(self, x, y, z, theta, phi, beta):
        self.x = x
        self.y = y
        self.z = z
        self.theta = theta
        self.phi = phi
        self.beta = beta
        self.calculate_zeroth_plane_params()
    def get_pos(self):
        return np.array([self.x, self.y, self.z])
    def calculate_zeroth_plane_params(self):
        normal = np.array([np.cos(self.phi),
                  0,
                  -np.sin(self.phi)])
        self.zeroth_plane = plane(normal[0], 
                                  normal[1], 
                                  normal[2], 
                                  np.dot(normal, self.get_pos()))
    def get_nth_plane_params(self, n):
        if self.zeroth_plane.d < 0:
            n = -n # flip sign of n if d points towards the origin.
        return self.zeroth_plane.params + [0, 0, 0, n*self.delta_pixel]
    def calculate_all_n_plane_params(self, n_pos, n_neg):
        # calculates equations for planes deviating by n from the zeroth plane.
        self.planes = []
        for n in range(-n_neg, n_pos + 1):
            self.planes.append(plane(*self.get_nth_plane_params(n)))
        return self.planes
    def calculate_delta_pixel(self, focal_len):
        self.delta_pixel = (self.beta * focal_len) / PIX_WIDTH
        return self.delta_pixel


class camera():
    # holds various camera parameters.
    def __init__(self, focal_len, x_c, y_c):
        self.focal_len = focal_len
        self.x_c = round(x_c)
        self.y_c = round(y_c)
        self.optic_centre = np.array([self.x_c, self.y_c])



def calculate_img_coordinates(pix_coords, camera, beta):
    # returns [x, y, z] coordinates of the point given by pix_coords
    # in terms of real-world distance. z will be the focal length.
    # in the equations, this is x_i, y_i, z_i.
    img_plane_width = beta * camera.focal_len # height not necessary. ratio is enough.
    centred_pix_coords = pix_coords - camera.optic_centre
    img_coords = centred_pix_coords * (img_plane_width/PIX_WIDTH)
    img_coords = np.append(img_coords, camera.focal_len)
    #####################################################
    # ISSUE HERE: Z COORD SHOULD BE NEGATIVE FOCAL LEN.
    # THIS WILL BE CORRECT BY DEFAULT IF X AND Y CONVENTION FIXED ELSEWHERE.
    #####################################################
    return img_coords


def calculate_depth_for_point(img_coords, nth_plane):
    # x_i, y_i, z_i = img_coords[0], img_coords[1], img_coords[2]
    # depth = (d*z_i) / (a*x_i + b*y_i + c*z_i)
    # denominator. represents light plane normal dotted with image pixel vector.
    r_i_dot_n = (nth_plane.a*img_coords[0] + nth_plane.b*img_coords[1] + nth_plane.c*img_coords[2])
    # numerator. represents light plane normal, dotted with projector
    # position vector, scaled by focal length (z_i).
    r_p_dot_n_z_i = (nth_plane.d*img_coords[2])
    # reject values with division by 0.
    if r_i_dot_n == 0:
        return np.sign(r_p_dot_n_z_i) * MAX_DEPTH
    
    depth = r_p_dot_n_z_i / r_i_dot_n
    if (depth * np.sign(depth)) >= MAX_DEPTH:
        # above threshold value; set to threshold.
        return np.sign(depth) * MAX_DEPTH
    else:
        return depth


def convert_indices_to_depths(indices, projector, camera):
    n_pos = PIX_WIDTH - camera.x_c
    n_neg = PIX_WIDTH - n_pos
    projector.calculate_delta_pixel(camera.focal_len)
    planes = projector.calculate_all_n_plane_params(n_pos, n_neg)
    # for 600 pixels wide, we can have 512 or 1024 indices with binary. code generates 1024,
    # but we can only use 600. so scale all indices accordingly.
    num_indices = 2*(np.ceil(np.log2(PIX_WIDTH)))
    scale = PIX_WIDTH / num_indices
    indices = np.round(indices * scale).astype(int) # round to nearest integer
    depth = []
    for i_row, row in enumerate(indices):
        depth.append([])
        for i_col, pix_index in enumerate(row):
            if pix_index >= PIX_WIDTH:
                nth_plane = planes[PIX_WIDTH - 1]
            else:
                nth_plane = planes[pix_index]
            img_coords = calculate_img_coordinates(np.array([i_col, i_row]), camera, projector.beta)
            depth[-1].append(round(calculate_depth_for_point(img_coords, nth_plane)))
    return depth



# proj = projector(X, Y, Z, THETA, PHI, BETA)
# cam = camera(FOCAL_LEN, X_C, Y_C)

import numpy as np
import cv2 as cv

from alive_progress import alive_bar

from general import *
from projcam import *



class sim_env():
    '''class that contains the simulation environment. includes info
    about points, lines, planes, and surfaces in the environment.'''
    def __init__(self):
        self.points = []
        self.lines = []
        self.planes = []
        self.surfaces = []
        self.surf_bounds = []
        self.surf_resolutions = []
        # self.ax = setup_3d()
    
    def add_points(self, points):
        for point in points:
            self.points.append(point)
    
    def add_lines(self, lines):
        for line in lines:
            self.lines.append(line)
        
    def add_planes(self, planes):
        for plane in planes:
            self.planes.append(plane)

    def add_surfaces(self, surfaces, surface_bounds, resolutions):
        for i, surface in enumerate(surfaces):
            self.surfaces.append(surface)
            self.surf_bounds.append(surface_bounds[i])
            self.surf_resolutions.append(resolutions[i])

    def plot_points(self):
        plot_points(self.ax, self.points)
    
    def plot_lines(self, colour=False, dashed=False):
        for line in self.lines:
            plot_line(self.ax, line, colour, dashed)

    def plot_planes(self):
        for plane in self.planes:
            plot_plane(self.ax, plane)

    def plot_surfaces(self):
        for i, surf in enumerate(self.surfaces):
            plot_surface(self.ax, surf, self.surf_bounds[i], self.surf_resolutions[i])

    def plot_pix_lines(self, cam_bounds, cam_res, with_colours=False):
        self.camera.plot_pix_lines(self.ax, cam_bounds, cam_res, with_colours, True)

    def plot_corner_lines(self, cam_bounds, cam_res, proj_bounds, proj_res, with_colours=True):
        self.camera.plot_corner_lines(self.ax, cam_bounds, cam_res, with_colours, True)
        self.projector.plot_corner_lines(self.ax, proj_bounds, proj_res, None, True)

    def plot_pix_planes(self):
        for plane in self.pcorners:
            for i_r, row in enumerate(plane):
                for i_c, corners in enumerate(row):
                    x = corners[0][0], corners[1][0], corners[2][0], corners[3][0]
                    y = corners[0][1], corners[1][1], corners[2][1], corners[3][1]
                    z = corners[0][2], corners[1][2], corners[2][2], corners[3][2]
                    try:
                        self.ax.plot_trisurf(x, y, z, alpha=0.5, color = self.projector.pix_info[i_r][i_c], triangles=[[0,1,2], [1,2,3]])
                    except OverflowError:
                        print("OverflowError. Skipping plane plotting.")
                        print("Planes/surfaces may be very large. Try scaling this down.")

    def plot_image_plane(self, scale=1):
        '''plots the camera-captured image on its corresponding image plane.
        this image plane is located a focal length away from the camera centre.'''
        t = self.camera.T
        focal_ray = self.camera.R @ np.array([0, 0, 1]) * self.camera.foc_len
        xpix_ray = self.camera.R @ np.array([1, 0, 0]) * self.camera.pix_len_x
        ypix_ray = self.camera.R @ np.array([0, 1, 0]) * self.camera.pix_len_y
        image = self.camera.captured_imgs[-1] # last image projected
        for i_r, row in enumerate(image):
            for i_c, pixel in enumerate(row):
                corn1 = focal_ray + xpix_ray * (i_c - 1/2 - self.camera.u0) + ypix_ray * (i_r - 1/2 - self.camera.v0)
                corn2 = focal_ray + xpix_ray * (i_c + 1/2 - self.camera.u0) + ypix_ray * (i_r - 1/2 - self.camera.v0)
                corn3 = focal_ray + xpix_ray * (i_c - 1/2 - self.camera.u0) + ypix_ray * (i_r + 1/2 - self.camera.v0)
                corn4 = focal_ray + xpix_ray * (i_c + 1/2 - self.camera.u0) + ypix_ray * (i_r + 1/2 - self.camera.v0)
                corn1 = t + corn1 * scale
                corn2 = t + corn2 * scale
                corn3 = t + corn3 * scale
                corn4 = t + corn4 * scale
                x = corn1[0], corn2[0], corn3[0], corn4[0]
                y = corn1[1], corn2[1], corn3[1], corn4[1]
                z = corn1[2], corn2[2], corn3[2], corn4[2]
                self.ax.plot_trisurf(x, y, z, alpha=0.5, color = pixel, triangles=[[0,1,3], [3,2,0]])

    def plot_R_Kinv_u(self, u, resolution, scale=1):
        '''plots camera pixel line u. returns sim_line object.'''
        t_c = self.camera.T
        frkinvu = self.camera.foc_len * self.camera.R @ self.camera.Kinv @ u * scale
        bbox = bounding_box([[min(t_c[0], frkinvu[0]), max(t_c[0], frkinvu[0])],
                             [min(t_c[1], frkinvu[1]), max(t_c[0], frkinvu[1])],
                             [min(t_c[2], frkinvu[2]), max(t_c[2], frkinvu[2])]])
        line = sim_line(t_c, frkinvu, bbox, resolution)
        plot_line(self.ax, line)
        return line


    def plot_intersections(self, cam_display_text=None, proj_display_text=None):
        '''plots all previously calculated pixel-line intersections
        for both projector and camera, including planes and surfaces.'''
        proj_ints = self.flatten_point_grid_array(self.proj_ints)
        cam_ints = self.flatten_point_grid_array(self.cam_pints)
        cam_ints += self.flatten_point_grid_array(self.cam_sints)
        plot_points(self.ax, proj_ints, proj_display_text)
        plot_points(self.ax, cam_ints, cam_display_text)

    def flatten_point_grid_array(self, array):
        '''takes an array with columns and rows and flattens it into a long list.'''
        flat = []
        for row in array:
            for item in row:
                flat.append(item)
        return flat

    def rotate_lines(self, R):
        '''rotates all lines in the simulation environment by the rotation matrix R.'''
        for line in self.lines:
            line.b = R.dot(line.b)

    def setup_projector(self, K, n_pix_width, n_pix_height):
        self.projector = sim_projector(K, n_pix_width, n_pix_height)

    def setup_camera(self, K, foc_len, n_pix_width, n_pix_height, R, T):
        self.camera = sim_camera(K, foc_len, n_pix_width, n_pix_height, R, T)

    def show_projcam_locations(self, labels=True):
        '''shows locations of the camera and projector as dots.'''
        t_p = self.projector.T
        t_c = self.camera.T
        self.ax.scatter([t_p[0], t_c[0]], [t_p[1], t_c[1]], [t_p[2], t_c[2]])
        if labels:
            self.ax.text(t_p[0], t_p[1], t_p[2], 'Projector', va='center', ha='center')
            self.ax.text(t_c[0], t_c[1], t_c[2], 'Camera', va='center', ha='center')

    def calc_projector_pixel_centres(self, line_resolution):
        '''projector pixel lines lie at the corners. this calculates their centre lines.'''
        self.proj_cints = self.projector.calc_pix_centre_lines(self, line_resolution)

    def show_light_plane(self, n, vertical=True, corner=False):
        '''shows the nth light-plane, i.e., the plane formed by the
        pixel line vectors of the nth row/column (depending on vertical).'''
        if corner:
            ints = self.proj_ints
        else:
            ints = self.proj_cints
        if vertical:
            top_corner = ints[0][0][n]
            bottom_corner = ints[0][-1][n]
            info = self.projector.projected_imgs[-1][0][n]
        else:
            top_corner = ints[0][n][0]
            bottom_corner = ints[0][n][-1]
            info = self.projector.projected_imgs[-1][n][0]
        x = top_corner[0], bottom_corner[0], self.projector.T[0]
        y = top_corner[1], bottom_corner[1], self.projector.T[1]
        z = top_corner[2], bottom_corner[2], self.projector.T[2]
        self.ax.plot_trisurf(x, y, z, alpha=0.5, color = info)

    def plot_normal_vector(self, n, scale=1):
        '''plots the normal vector n.'''
        t_p = self.projector.T
        t_n = t_p + scale * n
        bbox = bounding_box([[min(t_p[0], t_n[0]), max(t_p[0], t_n[0])],
                             [min(t_p[1], t_n[1]), max(t_p[0], t_n[1])],
                             [min(t_p[2], t_n[2]), max(t_p[2], t_n[2])]])
        line = sim_line(self.projector.T, n, bbox, 2)
        plot_line(self.ax, line)

    def show_camera_translation(self):
        t_p = self.projector.T
        t_c = self.camera.T
        bbox = bounding_box([[min(t_p[0], t_c[0]), max(t_p[0], t_c[0])],
                             [min(t_p[1], t_c[1]), max(t_p[0], t_c[1])],
                             [min(t_p[2], t_c[2]), max(t_p[2], t_c[2])]])
        trans = sim_line(t_p, t_c, bbox, 2)
        plot_line(self.ax, trans)

    def calc_proj_pix_indices_for_camera_grid(self, line_resolution, show_progress=False):
        # calculate pixel lines
        self.projector.calc_pix_lines()
        self.camera.calc_pix_lines()
        if show_progress: print("Pixel lines calculated.")

        # calculate intersections of pixel lines with surfaces in sim_env
        self.proj_ints = self.projector.calc_intersections(self, line_resolution)
        if show_progress: print("Intersections calculated.")

        # find the corners of projector pixel planes on surfaces
        self.pcorners = self.projector.get_four_corners(self.proj_ints)
        self.ppix_planes = self.projector.get_all_pix_planes(self.pcorners)
        if show_progress: print("Corners calculated.")

        # calculate closest projector intersection for each camera intersection.
        point_pix_coords = [] # nearest pixel coordinate for each intersection.
        # method for when using least squares. somewhat temperamental.
        # with alive_bar(len(self.camera.pix_lines)) as row_bar:
        #     for cam_row in self.camera.pix_lines:
        #         point_pix_coords.append([])
        #         for cam_line in cam_row:
        #             point_pix_coords[-1].append([None])
        #             # if multiple plane-line intersections are found, the one with
        #             # the smallest distance from the camera is used.
        #             min_cam_dist = np.inf
        #             # iterate over all planes
        #             for i_r, row in enumerate(self.ppix_planes):
        #                 for i_c, planes in enumerate(row):
        #                     for i_p, plane_params in enumerate(planes):
        #                         intsec, point = self.check_plane_line_intersection(plane_params, cam_line)
        #                         if intsec:
        #                             dist = np.linalg.norm(point - self.camera.T)
        #                             if dist < min_cam_dist:
        #                                 min_cam_dist = dist
        #                                 point_pix_coords[-1][-1] = [i_r, i_c]
        #         row_bar() # progress indicator.
        with alive_bar(len(self.camera.pix_lines)) as row_bar:
            for cam_row in self.camera.pix_lines:
                point_pix_coords.append([])
                for cam_line in cam_row:
                    point_pix_coords[-1].append([None])
                    # if multiple plane-line intersections are found, the one with
                    # the smallest distance from the camera is used.
                    min_cam_dist = np.inf
                    # iterate over all planes
                    for i_r, row in enumerate(self.ppix_planes):
                        for i_c, planes in enumerate(row):
                            for plane_params in planes:
                                intsec, point = self.check_plane_line_intersection(plane_params, cam_line)
                                if intsec:
                                    # if i_r == 2 and i_c == 2:
                                    #     self.ax.scatter(point[0], point[1], point[2], c='orange')
                                    dist = np.linalg.norm(point - self.camera.T)
                                    if dist < min_cam_dist:
                                        min_cam_dist = dist
                                        point_pix_coords[-1][-1] = [i_r, i_c]
                row_bar() # progress indicator.

        self.proj_cam_indices = point_pix_coords

    def set_proj_info(self, pix_info):
        self.projector.assign_pix_info(pix_info)

    def get_cam_info(self, null_info=np.array([0.1, 0.1, 0.1])):
        '''returns information seen by the camera.'''
        pix_info = self.projector.get_pix_info()
        cam_info = []
        for row in self.proj_cam_indices:
            cam_info.append([])
            for index in row:
                if index[0] is None:
                    cam_info[-1].append(null_info)
                else:
                    cam_info[-1].append(pix_info[index[0]][index[1]])
        if len(cam_info) == 0:
            print("Error: proj_cam_indices not set.")
        else:
            self.camera.assign_pix_info(cam_info)
            self.camera.captured_imgs.append(cam_info)
            return np.array(cam_info)
    
    def show_info_img(self, proj_or_cam='cam', new_size=None, convert_rgb=True):
        '''displays an image stored in the camera or the projector,
        depending on the setting. defaults to camera.'''
        if proj_or_cam == 'cam':
            info = self.camera.get_pix_info()
        elif proj_or_cam == 'proj':
            info = self.projector.get_pix_info()
        if convert_rgb == True: # swap bgr to rgb
            for row in info:
                for pixel in row:
                    pixel[0], pixel[2] = pixel[2], pixel[0]
        if new_size is not None: # resize
            info = cv.resize(info, new_size, interpolation = cv.INTER_NEAREST)
        cv.imshow(proj_or_cam, info)
        
    def check_plane_line_intersection(self, plane, line, least_squares=False):
        '''takes sim_plane and sim_line objects and checks if the line
        strikes the plane within the plane bounding box. returns bool, x.'''
        # check that b.n =/= 0 as this means plane and line are parallel.
        if least_squares:
            intersects, x = self.check_subplane_intersection(plane, line)
        else:
            intersects = False
            x = None
            for sub_plane in plane:
                sub_ints, sub_x = self.check_subplane_intersection(sub_plane, line)
                if sub_ints:
                    intersects = True
                    x = sub_x
                    break
        return intersects, x
        
    def check_subplane_intersection(self, plane, line):
        '''takes a subplane defined by corners, and a line, and determines
        if the two intersects within the bounds of the corners. returns bool, x.'''
        bn = plane.n.dot(line.b)
        if bn == 0:
            print("Error: plane and line are parallel.")
        else:
            an = plane.n.dot(line.a)
            lamb = (plane.d - an) / bn
            x = line.a + lamb * line.b
            # if x within bbox, then x is not removed -> len=1. else len=0.
            within_bb = bool(len(plane.bounds.remove_external_points([x])))
            if within_bb:
                # check orientation of n
                if plane.n.dot(x) > 0:
                    multiplier = -1
                else:
                    multiplier = 1

                # find lines tangent to pixel edges
                edge_lines = []
                for line in plane.pcorner_lines:
                    edge_lines.append(np.cross(line, multiplier * plane.n))

                # find lines on plane from corners to x
                l_cornx = []
                for corner in plane.corners:
                    l_cornx.append(x - corner)

                for i_c, line in enumerate(edge_lines):
                    if l_cornx[i_c].dot(line) < 0:
                        within_bb = False
                        break
            return within_bb, x

        
    
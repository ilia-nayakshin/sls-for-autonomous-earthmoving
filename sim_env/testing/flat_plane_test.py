import sys, os
import numpy as np
import cv2 as cv

# for finding correct local modules
sys.path.append(os.path.abspath(os.path.join('.', 'sim_env')))

from envir import sim_env
from envir_objects import bounding_box
from general import global_vars, project_and_capture, euler_rotation_matrix

from test_funcs import test_plane
from errors import find_img_sq_error


def testcase_flat_plane():
    '''A test where camera and projector are normal to a flat plane,
    with the two in the same location.'''
    # learning rate should be 0.01.
    # tolerance 1e-8.

    gvs = global_vars() # get global variables
    plane = test_plane(100, np.array([1, 0, 0])) # define plane function
    bbox = bounding_box([[0, 110], [-40, 40], [-40, 40]]) # define bounds
    cambbox = bounding_box([[0, 110], [-40, 40], [-40, 40]])

    # setup simulation environment
    sim = sim_env()
    sim.add_surfaces([plane], [bbox], [100])

    # redefine T and R as appropriate.
    gvs.T = np.zeros(3)
    gvs.R = np.eye(3)

    # project, capture, and find error.
    # 0 displacement, 0 rotation.
    # sim = project_and_capture(sim, gvs, bbox, cambbox, True, False, True, True, False, False)
    # find_img_sq_error(sim.camera.get_pix_info(), sim.projector.get_pix_info(), gvs.SHOW_IMG_SIZE, True)


    # TAGGING ON DISPLACEMENT TESTS
    # +ve y displacement
    # gvs.T = np.array([0, 10, 0])
    # sim = project_and_capture(sim, gvs, bbox, cambbox, False, False, False, False, False, False)
    # expected = sim.projector.get_pix_info()
    # middle = int(np.ceil(len(expected)/2))
    # expected[:, middle:] = expected[:, :middle]
    # expected[:, :middle] = np.zeros(np.shape(expected[:, :middle]))
    # find_img_sq_error(sim.camera.get_pix_info(), expected, gvs.SHOW_IMG_SIZE, True)

    # # -ve y displacement
    # gvs.T = np.array([0, -10, 0])
    # sim = project_and_capture(sim, gvs, bbox, cambbox, False, False, False, False, False, False)
    # expected = sim.projector.get_pix_info()
    # middle = int(np.ceil(len(expected)/2))
    # expected[:, :middle] = expected[:, middle:]
    # expected[:, middle:] = np.zeros(np.shape(expected[:, middle:]))
    # find_img_sq_error(sim.camera.get_pix_info(), expected, gvs.SHOW_IMG_SIZE, True)

    # # +ve z displacement
    # gvs.T = np.array([0, 0, 10])
    # sim = project_and_capture(sim, gvs, bbox, cambbox, False, False, False, False, False, False)
    # expected = sim.projector.get_pix_info()
    # middle = int(np.ceil(len(expected)/2))
    # expected[middle:] = expected[:middle]
    # expected[:middle] = np.zeros(np.shape(expected[:middle]))
    # find_img_sq_error(sim.camera.get_pix_info(), expected, gvs.SHOW_IMG_SIZE, True)

    # # -ve z displacement
    # gvs.T = np.array([0, 0, -10])
    # sim = project_and_capture(sim, gvs, bbox, cambbox, False, False, False, False, False, False)
    # expected = sim.projector.get_pix_info()
    # middle = int(np.ceil(len(expected)/2))
    # expected[:middle] = expected[middle:]
    # expected[middle:] = np.zeros(np.shape(expected[middle:]))
    # find_img_sq_error(sim.camera.get_pix_info(), expected, gvs.SHOW_IMG_SIZE, True)

    # +ve y, -ve z displacement
    # gvs.T = np.array([0, 10, -10])
    # sim = project_and_capture(sim, gvs, bbox, cambbox, False, False, False, False, False, False)
    # expected = sim.projector.get_pix_info()
    # middle = int(np.ceil(len(expected)/2))
    # expected[:middle, middle:] = expected[middle:, :middle]
    # expected[middle:] = np.zeros(np.shape(expected[middle:]))
    # expected[:, :middle] = np.zeros(np.shape(expected[:, :middle]))
    # find_img_sq_error(sim.camera.get_pix_info(), expected, gvs.SHOW_IMG_SIZE, True)

    # +ve x displacement
    # gvs.T = np.array([50, 0, 0])
    # sim = project_and_capture(sim, gvs, bbox, cambbox, True, False, False, False, False, False)
    # projected = sim.projector.get_pix_info()
    # low_corn = int(np.ceil(len(projected)/4))
    # high_corn = 3 * low_corn
    # expected = projected[low_corn:high_corn, low_corn:high_corn]
    # expected = cv.resize(expected, np.shape(projected)[:2], interpolation = cv.INTER_NEAREST)
    # find_img_sq_error(sim.camera.get_pix_info(), expected, gvs.SHOW_IMG_SIZE, True)

    # -ve x displacement
    # gvs.T = np.array([-100, 0, 0])
    # gvs.CPIX_NUM = gvs.PPIX_NUM*2 # to capture all projected pixels well
    # sim = project_and_capture(sim, gvs, bbox, cambbox, True, False, False, False, False, False)
    # projected = sim.projector.get_pix_info()
    # low_corn = int(np.ceil(gvs.CPIX_NUM/4))
    # high_corn = 3 * low_corn
    # expected = np.zeros((gvs.CPIX_NUM, gvs.CPIX_NUM, 3))
    # expected[low_corn:high_corn, low_corn:high_corn] = projected
    # find_img_sq_error(sim.camera.get_pix_info(), expected, gvs.SHOW_IMG_SIZE, True)

    # -ve x, +ve y, -ve z displacement
    # gvs.T = np.array([-100, 10, -10])
    # gvs.CPIX_NUM = gvs.PPIX_NUM*2 # to capture all projected pixels well
    # sim = project_and_capture(sim, gvs, bbox, cambbox, True, False, False, False, False, False)
    # projected = sim.projector.get_pix_info()
    # middle = int(np.ceil(gvs.CPIX_NUM/2))
    # expected = np.zeros((gvs.CPIX_NUM, gvs.CPIX_NUM, 3))
    # expected[:middle, middle:] = projected
    # find_img_sq_error(sim.camera.get_pix_info(), expected, gvs.SHOW_IMG_SIZE, True)

    # T=0, rotation in +ve z direction
    # gvs.T = np.zeros(3)
    # gvs.R = euler_rotation_matrix(0.1, 0, 0)
    # gvs.CPIX_NUM = gvs.PPIX_NUM
    # sim = project_and_capture(sim, gvs, bbox, cambbox, True, False, True, False, False, True)
    # expected = sim.projector.get_pix_info()
    # middle = int(np.ceil(len(expected)/2))
    # expected[:, middle:] = expected[:, :middle]
    # expected[:, :middle] = np.zeros(np.shape(expected[:, :middle]))
    # find_img_sq_error(sim.camera.get_pix_info(), expected, gvs.SHOW_IMG_SIZE, True)

    # T=0, rotation in -ve z direction
    # gvs.T = np.zeros(3)
    # gvs.R = euler_rotation_matrix(-0.1, 0, 0)
    # gvs.CPIX_NUM = gvs.PPIX_NUM
    # sim = project_and_capture(sim, gvs, bbox, cambbox, True, False, True, False, False, True)
    # expected = sim.projector.get_pix_info()
    # middle = int(np.ceil(len(expected)/2))
    # expected[:, :middle] = expected[:, middle:]
    # expected[:, middle:] = np.zeros(np.shape(expected[:, middle:]))
    # find_img_sq_error(sim.camera.get_pix_info(), expected, gvs.SHOW_IMG_SIZE, True)

    # T=0, rotation in +ve y direction
    # gvs.T = np.zeros(3)
    # gvs.R = euler_rotation_matrix(0, 0.1, 0)
    # gvs.CPIX_NUM = gvs.PPIX_NUM
    # sim = project_and_capture(sim, gvs, bbox, cambbox, True, False, True, False, False, True)
    # expected = sim.projector.get_pix_info()
    # middle = int(np.ceil(len(expected)/2))
    # expected[:middle] = expected[middle:]
    # expected[middle:] = np.zeros(np.shape(expected[middle:]))
    # find_img_sq_error(sim.camera.get_pix_info(), expected, gvs.SHOW_IMG_SIZE, True) 

    # T=0, rotation in -ve y direction
    # gvs.T = np.zeros(3)
    # gvs.R = euler_rotation_matrix(0, -0.1, 0)
    # gvs.CPIX_NUM = gvs.PPIX_NUM
    # sim = project_and_capture(sim, gvs, bbox, cambbox, True, False, True, False, False, True)
    # expected = sim.projector.get_pix_info()
    # middle = int(np.ceil(len(expected)/2))
    # expected[middle:] = expected[:middle]
    # expected[:middle] = np.zeros(np.shape(expected[:middle]))
    # find_img_sq_error(sim.camera.get_pix_info(), expected, gvs.SHOW_IMG_SIZE, True)

    # T=0, rotation in +ve y direction, beta small.
    gvs.T = np.zeros(3)
    gvs.BETA = 0.1
    gvs.R = euler_rotation_matrix(0, -gvs.BETA, 0)
    gvs.CPIX_NUM = gvs.PPIX_NUM
    sim = project_and_capture(sim, gvs, bbox, cambbox, True, False, True, True, False, True)
    expected = sim.projector.get_pix_info()
    middle = int(np.ceil(len(expected)/2))
    expected[middle:] = expected[:middle]
    expected[:middle] = np.zeros(np.shape(expected[:middle]))
    find_img_sq_error(sim.camera.get_pix_info(), expected, gvs.SHOW_IMG_SIZE, True)

    # T=0, rotation in +ve y direction, beta large.
    gvs.T = np.zeros(3)
    gvs.R = euler_rotation_matrix(0, -0.3, 0)
    gvs.BETA = 0.3
    gvs.CPIX_NUM = gvs.PPIX_NUM
    sim = project_and_capture(sim, gvs, bbox, cambbox, True, False, True, True, False, True)
    expected = sim.projector.get_pix_info()
    middle = int(np.ceil(len(expected)/2))
    expected[middle:] = expected[:middle]
    expected[:middle] = np.zeros(np.shape(expected[:middle]))
    find_img_sq_error(sim.camera.get_pix_info(), expected, gvs.SHOW_IMG_SIZE, True)    

testcase_flat_plane()
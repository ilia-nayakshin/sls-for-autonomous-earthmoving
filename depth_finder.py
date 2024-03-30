from calibration import calibrate_camera
from depth_calculation_functions import projector, camera, convert_indices_to_depths
from structured_light_testing import project_and_decode
from visualiser import plot_depths
import numpy as np
import cv2 as cv


# constants
X, Y, Z = 200, 0, 0 # mm
# THETA = 30 * (2*np.pi/360) # radians
THETA = 0 # radians, rotation around x
PHI = 0 # radians, rotation around y
BETA = 144/205 # projector ratio (img width/projection distance)
FOCAL_LEN = 4.44 # mm
PIX_WIDTH, PIX_HEIGHT = 960, 540
PIX_LEN = 1.4 * 10^(-6) * (3840 / PIX_WIDTH) # scale up if running at lower resolution

NUM_MENU_OPTIONS = {"calibration": 4}


def print_calibration_menu():
    print('----------')
    print('Calibration Stage.')
    print('')
    print('Select a way to calibrate:')
    print('1. Use values in camera.csv matrix.')
    print('2. Enter a matrix manually.')
    print('3. Calibrate from scratch.')
    print('4. None - Exit.')


def print_save_matrix_menu():
    print('----------')
    print('Camera matrix OK.')
    print('')
    print('Would you like to save this matrix? (y/n)')
    print('Saving will override any existing camera.csv.')


def menu_validation(value, max_num):
    # validates that the value is at most max_num (and various other things).
    if len(value) == 0 or len(value) > len(str(max_num)):
        print("Invalid input (length). Try again.")
        return False
    if not value.isdigit():
        print("Invalid input (not a number). Try again.")
        return False
    if int(value) > max_num or int(value) < 1:
        print("Invalid input (out of range). Try again.")
        return False
    return True


def y_n_validation(value):
    value = value.lower()
    if value != 'y' and value != 'n':
        print("Invalid input (not 'y' or 'n'). Try again.")
        return False
    return True


def find_calibration_matrix():
    # sets up the calibration matrix by finding a previous
    # matrix, allowing the user to manually enter a matrix, or
    # calibrating the camera from scratch.
    
    # repeating loop to exit.
    success = False
    while not success:
        print_calibration_menu()
        # choice = input(">>> ")
        choice = '1'
        valid = menu_validation(choice, NUM_MENU_OPTIONS["calibration"])
        if valid:
            choice = int(choice)
            if choice == 1:
                try:
                    # camera_matrix = np.array(list(csv.reader(open("camera.csv", "rb"), delimiter=","))).astype("int64")
                    camera_matrix = np.loadtxt(open("camera.csv", "rb"), delimiter=",", skiprows=0)
                    success = True
                except FileNotFoundError:
                    print("File not found. Try a different method.")
            elif choice == 2:
                pass # let user enter matrix manually.
                success = True
                save_camera_matrix(camera_matrix)
            elif choice == 3:
                camera_matrix = calibrate_camera()[1]
                success = True
                save_camera_matrix(camera_matrix)
            elif choice == 4:
                break
    if success == True:
        return camera_matrix
    else:
        return None


def save_camera_matrix(camera_matrix):
    # asks user if they would like to save a camera matrix,
    # overrides on yes.
    success = False
    while not success:
        print_save_matrix_menu()
        save = input('>>> ').lower()
        valid = y_n_validation(save)
        if valid:
            if save == 'y':
                np.savetxt("camera.csv", camera_matrix, delimiter=",")
            success = True # regardless of y or n.


camera_matrix = find_calibration_matrix()
proj = projector(X, Y, Z, THETA, PHI, BETA)
cam = camera(FOCAL_LEN, camera_matrix[0][2], camera_matrix[1][2])
# cam = camera(FOCAL_LEN, 0, 0)
# indices = project_and_decode()
indices = np.loadtxt(open("indices.csv", "rb"), delimiter=",", skiprows=1)
print('huh.')
depths = convert_indices_to_depths(indices, proj, cam)
# np.savetxt("indices.csv", indices, delimiter=",")
# np.savetxt("depths.csv", depths, delimiter=",")
# plot_depths(indices)
plot_depths(depths)
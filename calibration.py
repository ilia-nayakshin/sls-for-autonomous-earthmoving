# code modified from the docs website, with very significant alterations.
import numpy as np
import cv2 as cv
import ssl
import urllib.request


# constants.
SQUARE_SIZE = 28.5 # mm
GRID_ROWS = 6
GRID_COLS = 5
SUBPIX_DIM = 11 # 1/2 sidelength of search window for subpixel refinement.
NUM_CAL_IMGS_NEEDED = 20
REFRESH_RATE = 300 # ms
STREAM_URL = 'http://10.249.167.20:8080/shot.jpg'
IMG_WIDTH, IMG_HEIGHT = 960, 540 # half the height and width of the full-size image.
# termination criteria
CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)



def prepare_standard_objp_array():
    # generates a grid of object points based on the constants defined at the start of the program.
    objp = np.zeros((GRID_COLS*GRID_ROWS,3), np.float32)
    objp[:,:2] = np.mgrid[0:GRID_ROWS,0:GRID_COLS].T.reshape(-1,2) * SQUARE_SIZE
    return objp


def setup_stream():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE


def update_video_frame(camera_object, frame_name, press_key=32, esc_key=27, refresh_rate=1):
    # takes camera object to read frame from and updates window with frame_name
    # to have this image. returns image regardless of taken.
    # if escape key is hit, returns none.
    # pass frame name as none to not have a window visible.

    if camera_object == 'stream':
        # on stream setup, get image from webpage.
        img_response = urllib.request.urlopen(STREAM_URL)
        frame_array = np.array(bytearray(img_response.read()), dtype=np.uint8)
        frame = cv.imdecode(frame_array, -1)
        frame = cv.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
        ret = True
    else:
        # otherwise, get image from camera_object.
        ret, frame = camera_object.read()

    # check frame was grabbed correctly, display image.
    if not ret:
        print("failed to grab frame")
    else:
        if frame_name is not None:
            cv.imshow(frame_name, frame)

    # wait a number of milliseconds for a key to be pressed.
    key = cv.waitKey(refresh_rate)
    if key % 256 == esc_key:
        return None, None
    elif key % 256 == press_key:
        return True, frame
    else:
        return False, frame
    

def find_and_plot_chessboard(images, frame_name, initial_obj_points, image_points=[], obj_points=[]):
    # takes list of images and iterates through, pausing on each that finds a chessboard
    # and displaying the board. returns [bool] containing whether the board was found or not.
    chessboard_located = []
    for image in images:
        # computer vision works best with intensity values, not colours
        grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(grey, (GRID_ROWS, GRID_COLS), None)
        # if chessboard found, draw corners and pause for 500ms.
        if ret == True:
            # print("chessboard found successfully.")
            obj_points.append(initial_obj_points)
            refined_corners = cv.cornerSubPix(grey, corners, (SUBPIX_DIM, SUBPIX_DIM), (-1,-1), CRITERIA)
            image_points.append(refined_corners)
            # Draw and display the corners
            cv.drawChessboardCorners(image, (GRID_ROWS, GRID_COLS), refined_corners, ret)
            cv.imshow(frame_name, image)
            cv.waitKey(500)
        chessboard_located.append(ret)
    return chessboard_located, obj_points, image_points


def take_n_calibration_images(n, press_key=32, esc_key=27, refresh_rate=1, camera_index=0):
    # shows video, taking photo on press of press_key, closing on esc_key,
    # then attempts to identify a chessboard in each image taken. if found,
    # it adds it to the list. once n images with chessboards have been found,
    # the program exits, returning images on exit.

    # prepare video window
    if camera_index == 'stream':
        setup_stream()
        camera = 'stream'
    else:
        camera = cv.VideoCapture(camera_index)
    cv.namedWindow("camera")

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = prepare_standard_objp_array()

    images = []
    running = True
    while running:
        taken, image = update_video_frame(camera, "camera", press_key, esc_key, refresh_rate)
        if taken is None:
            # escape on escape key
            # print("escape key pressed, exiting.")
            running = False
        else:
            # add image to list
            # print("image taken.")
            found, img_points, obj_points = find_and_plot_chessboard([image], "camera", objp)
            found = found[0]
            if found:
                # print("chessboard successfully detected, adding image.")
                images.append(image)
            else:
                # print("no chessboard found, continuing.")
                pass
        
        # end loop if enough images taken
        if len(images) >= n:
            # print("taken all images, exiting.")
            running = False
    return images, img_points, obj_points


def calibrate_camera():
    # calculates intrinsic camera properties by taking several images of a grid.
    # for camera matrix, use the second item.
    # get successful calibration images
    images, obj_points, img_points = take_n_calibration_images(NUM_CAL_IMGS_NEEDED, refresh_rate=REFRESH_RATE, camera_index='stream')
    if len(images) >= NUM_CAL_IMGS_NEEDED:
        # images captured completed successfully; calibrate camera
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, images[0].shape[:2], None, None)
        return ret, mtx, dist, rvecs, tvecs
    else:
        # calibration failed due to user pressing esc button.
        print("calibration failed: exited prematurely.")




cv.destroyAllWindows()
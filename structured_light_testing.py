import structuredlight as sl  # generates arrays of structured light images
import numpy as np  # converts arrays to images
import cv2 as cv
from calibration import update_video_frame, setup_stream

# constants
WIDTH  = 640
HEIGHT = 480


def project_imgs_and_get_photos(images, ret_colours):
    cv.namedWindow('projecting', cv.WND_PROP_FULLSCREEN)
    cv.setWindowProperty("projecting", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    cv.setWindowProperty('projecting', cv.WND_PROP_TOPMOST, 1)
    setup_stream()

    photos = []
    coloured_photos = []
    # project images and capture photos
    for image in images:
        cv.imshow('projecting', image) # project stl image
        cv.waitKey(100) # wait 50ms. also does this in next line, so taking photo in middle of pause.
        _, photo = update_video_frame('stream', None, refresh_rate=500, press_key=27)  # take photo

        # make greyscale (stl is greyscale)
        grey_photo = cv.cvtColor(photo, cv.COLOR_BGR2GRAY)
        photos.append(grey_photo)  # add to list of photos for later
        coloured_photos.append(photo)
    cv.destroyAllWindows()

    if ret_colours:
        return photos, coloured_photos
    else:
        return photos


def project_and_decode(ret_colours=False):
    # setup images to project
    gray = sl.Gray()

    image_patterns = gray.generate((WIDTH, HEIGHT))
    image_list = [np.array(array) for array in image_patterns]
    image_list.insert(0, np.zeros(np.shape(image_list[0])))

    # take photos and decode them to indices
    photos = project_imgs_and_get_photos(image_list, ret_colours)
    if ret_colours:
        photos, coloured_photos = photos[0], photos[1]
    cv.namedWindow('photoviewer')
    cv.setWindowProperty('photoviewer', cv.WND_PROP_TOPMOST, 1)
    for photo in photos:
        cv.imshow('photoviewer', photo)
        cv.waitKey(500)
    if ret_colours:
        cv.imshow('photoviewer', coloured_photos[0])
        cv.waitKey(5000)
    cv.destroyAllWindows()

    # only use photos from index 1 and onwards
    # because photo 1 has no image projected in it.
    img_index = gray.decode(photos[1:], thresh=127)
    return img_index, coloured_photos[0]

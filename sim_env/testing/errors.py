import numpy as np
import cv2 as cv



def find_img_sq_error(result, prediction, new_size=None, error_img=False, convert_rgb=True):
    '''Takes the RBG images of result and prediction and
    calculates the normalised squared error from one to another.
    If error_img is True, shows an image of the error.'''
    if np.shape(result) != np.shape(prediction):
        prediction = cv.resize(prediction, np.shape(result)[:2], interpolation = cv.INTER_NEAREST)

    if convert_rgb: # swap bgr to rgb
        for row in result:
            for pixel in row:
                pixel[0], pixel[2] = pixel[2], pixel[0]
        for row in prediction:
            for pixel in row:
                pixel[0], pixel[2] = pixel[2], pixel[0]
    
    # calculate errors
    error = result - prediction
    squared_error = np.zeros(np.shape(result))
    total_error = 0
    for i_r, row in enumerate(error):
        for i_c, pix in enumerate(row):
            for i_rgb, colour in enumerate(pix):
                sq_err = colour**2
                squared_error[i_r, i_c, i_rgb] = sq_err
                total_error += sq_err

    # resize image for viewing
    if new_size is not None:
        squared_error = cv.resize(squared_error, new_size, interpolation = cv.INTER_NEAREST)
        result = cv.resize(result, new_size, interpolation = cv.INTER_NEAREST)
        prediction = cv.resize(prediction, new_size, interpolation = cv.INTER_NEAREST)

    # normalise total by number of pixels
    norm_tot_error = total_error / (np.shape(result)[0] * np.shape(result)[1])
    norm_error = squared_error / squared_error.max()

    # handle error images
    if error_img: 
        cv.imshow('Error over Image', norm_error)
        cv.waitKey(0)

    return norm_tot_error, error
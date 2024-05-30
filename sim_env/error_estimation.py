import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt



def estimate_depth_error(points, surf_func, neglect_zeros=True):
    '''takes surface function and calculated depth points, estimates
    the squared error, i.e., deviation of the points from the surface.
    returns total error, mean per-point error, and array of errors
    for each point in points. neglect_zeros = True means that any points
    at the origin are ignored for the purposes of error estimation.'''
    total_error = 0
    # total_diff = 0
    count = 0
    errors = np.zeros(np.shape(points)[:2])
    for i_r, row in enumerate(points):
        for i_c, point in enumerate(row):
            x_s, y_s, z_s = surf_func(point[0], point[1], point[2])
            surf_point = np.array([x_s[0][0], y_s[0][0], z_s[0][0]])
            diff = surf_point.T - point
            error = np.sqrt(np.dot(diff, diff))
            if (neglect_zeros and np.linalg.norm(point) != 0.0) or not neglect_zeros:
                # total_diff += diff
                total_error += error
                errors[i_r, i_c] = error
                count += 1
    if count != 0:
        mean_error = total_error / count
        # mean_diff = total_diff / count
    else:
        mean_error = None
        # mean_diff = None
    return total_error, mean_error, errors
 


def show_error_img(error, resolution):
    '''shows an image of size 'resolution' with the image intensity
    indicating error. brighter pixels means higher error.'''
    # normalise pixels by peak value
    peak = error.max()
    image = cv.resize(error / peak, resolution, interpolation = cv.INTER_NEAREST)
    cv.imshow('error', image)


def show_error_bar_chart(error, hide_zero_bars=True):
    '''displays a 3d bar chart of the errors across the pixels. higher
    bar means higher error. requires plt.show() call following it.'''
    # setup x, y
    xx, yy = np.meshgrid(np.arange(len(error[0])), np.arange(len(error)))
    x, y = xx.ravel(), yy.ravel()
    error = error.ravel()
    # remove zero-bar floor. useful if useful information is surrounded by empty data.
    indices = []
    if hide_zero_bars:
        for i in range(len(error) - 1, -1, -1):
            if error[i] == 0.0:
                indices.append(i)
        error = np.delete(error, indices)
        x = np.delete(x, indices)
        y = np.delete(y, indices)
    # setup figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.bar3d(x, y, np.zeros_like(x), 1, 1, error, shade=True)
    ax.set_title("Error")
    ax.set_xlabel('x coordinate (pixels)')
    ax.set_ylabel('y coordinate (pixels)')
    ax.set_zlabel('Error Magnitude')
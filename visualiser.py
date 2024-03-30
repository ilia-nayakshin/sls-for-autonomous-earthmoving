import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import rgb2hex
from matplotlib import cm
from scipy.stats import norm


# constants
PROB_BOUND = 0.005  # reject outliers with less than x% probability.


def plot_depths(depths, x=None, y=None, flip_y=True):
    # plots depths as a 3d mesh. depths should be a matrix
    # containing the depths for each pixel, in mm.
    # can pass in x and y, if you want to reduce the resolution.
    # default is plotting all data points.
    # assumes gaussian distribution and plots only mean +- std * 3.
    
    # set up x and y if they are not passed in.
    if x is None:
        x = np.arange(len(depths[0]))
        if len(x) == 1:
            # rows of numpy matrices can register as length 1.
            try:
                x = np.arange(depths.shape[0])
            except AttributeError:
                x = np.arange(len(depths[0]))
    if y is None:
        y = np.arange(len(depths))
        if flip_y:
            y = np.flip(y)
        # x and y both need to be matrices. the same vector over and over.
        x, y = np.meshgrid(x, y)
    elif flip_y:
        y = np.flip(y)

    depths = np.matrix(depths)

    # calculate gaussian mean and std for threholding and min/max
    mean_x, std_x = compute_gaussian_params(x)
    mean_y, std_y = compute_gaussian_params(y)
    mean_z, std_z = compute_gaussian_params(depths)

    # threshold values
    thresholded_x = reject_outliers(mean_x, std_x, PROB_BOUND, x)
    thresholded_y = reject_outliers(mean_y, std_y, PROB_BOUND, y)
    thresholded_z = reject_outliers(mean_z, std_z, PROB_BOUND, depths)

    # setup plot.
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # Plot the surface.
    surf = ax.plot_surface(thresholded_x, thresholded_y, thresholded_z, 
                           cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # recompute mean and std for thresholded values - for min/max
    thr_mean_x, thr_std_x = compute_gaussian_params(thresholded_x)
    thr_mean_y, thr_std_y = compute_gaussian_params(thresholded_y)
    thr_mean_z, thr_std_z = compute_gaussian_params(thresholded_z)

    # comput min and max for thresholded values
    min_x, max_x = compute_min_max(thr_mean_x, thr_std_x)
    min_y, max_y = compute_min_max(thr_mean_y, thr_std_y)
    min_z, max_z = compute_min_max(thr_mean_z, thr_std_z)

    # change limits based on mins and maxes
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_zlim(min_z, max_z)

    # Add a color bar which maps values to colors
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def compute_gaussian_params(values):
    # assuming gaussian distribution, computes mean and std of values.

    # convert to np array if needed
    if type(values) is list:
        values = np.array(values)
    
    # flatten matrix to just values
    flattened_values = values.flatten()
    mean, std = np.nanmean(flattened_values), np.nanstd(flattened_values)
    return mean, std


def compute_min_max(mean, std, stdevs=3):
    # computes minimum and maximum values for confidence bounds
    # given by mean +- stdevs * std.
    min, max = (mean - std * stdevs), (mean + std * stdevs)
    return min, max


def reject_outliers(mean, std, prob_bound, values):
    # replaces outliers in values with 'nan'. if the gaussian probability of
    # observing a given value is less than the probability bound, the value
    # is rejected.

    # calculate value bounds
    lower_bound = norm.ppf(prob_bound, loc=mean, scale=std)
    upper_bound = norm.ppf(1 - prob_bound, loc=mean, scale=std)
    out_of_bounds = np.where((values < lower_bound) | (values > upper_bound), 0.0, 1.0)
    thresholded = np.multiply(out_of_bounds, values)
    thresholded[thresholded==0] = np.nan
    return thresholded


def coloured_scatter(x, y, z, colours, subsample_factor=1):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')

    colours = colours / 255
    # string array with length 7.
    hex_colours = np.zeros((np.shape(colours)[0], np.shape(colours)[1]), 'U7')
    for i_row in range(len(colours)):
        for i_pix in range(len(colours[0])):
            # opencv saves images as bgr, not rgb.
            hex_colours[i_row][i_pix] = rgb2hex(np.flip(colours[i_row][i_pix]))
    # colours = [[ rgb2hex(row[i_pix, :]) for i_pix in range(row.shape[0]) ] for row in colours]

    x = np.array(x).flatten()
    y = np.array(y).flatten()
    z = np.array(z).flatten()
    hex_colours = np.array(hex_colours).flatten()

    x = subsample_array(x, subsample_factor)
    y = subsample_array(y, subsample_factor)
    z = subsample_array(z, subsample_factor)
    hex_colours = subsample_array(hex_colours, subsample_factor)

    # calculate gaussian mean and std for threholding and min/max
    mean_x, std_x = compute_gaussian_params(x)
    mean_y, std_y = compute_gaussian_params(y)
    mean_z, std_z = compute_gaussian_params(z)

    # threshold values
    thresholded_x = reject_outliers(mean_x, std_x, PROB_BOUND, x)
    thresholded_y = reject_outliers(mean_y, std_y, PROB_BOUND, y)
    thresholded_z = reject_outliers(mean_z, std_z, PROB_BOUND, z)

    # ax.scatter(x, y, z, c=hex_colours)
    ax.scatter(thresholded_x, thresholded_y, thresholded_z, c=hex_colours)
    # for i in range(len(x)):
    #     for j in range(len(x[0])):
    #         ax.scatter(x[i][j], y[i][j], z[i][j], color=colours[i][j]/255)
        # if i % 60 == 0:
            # print(i, len(x))
        # print(i, len(x))
    plt.show()


def subsample_array(arr, subsample_factor):
    subsampled = []
    for i, item in enumerate(arr):
        if i % subsample_factor == 0:
            subsampled.append(item)
    return subsampled
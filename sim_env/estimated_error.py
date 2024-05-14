import numpy as np
import matplotlib.pyplot as plt

ZBAR = 100
L_S = 20
N = 40

X = 1000
Z = -1000



def calculate_bounds(n, x, z, zbar, l_s, n_pix):
    '''takes the position of the camera and the test surface (a plane)
    and a given pixel and calculates the theoretical error bounds for that pixel.'''
    # if odd, n_pix should be an integer. if odd, n_pix should be an integer +0.5.
    x_centre = n_pix * l_s
    x_lower = x_centre - l_s / (2 * n)
    x_upper = x_centre + l_s / (2 * n)
    a_1 = (z * (x - x_lower) + (zbar - z) * x) / ((x - x_lower) + (zbar - z) * (x_centre / zbar))
    a_2 = (z * (x - x_upper) + (zbar - z) * x) / ((x - x_upper) + (zbar - z) * (x_centre / zbar))
    error_1, error_2 = a_1 - zbar, a_2 - zbar
    return [error_1, error_2]


def calculate_all_bounds(n, x, z, zbar, l_s):
    end = (n - 1) / 2
    if end == 0:
        pixels = [0]
    else:
        pixels = np.arange(-end, end, 1)
    all_bounds = []
    for pix in pixels:
        all_bounds.append(calculate_bounds(n, x, z, zbar, l_s, pix))
    return all_bounds


def calc_expected_error(bounds):
    total = 0
    for bound in bounds:
        exp_error = (bound[0]**2 + bound[1]**2) / (2 * np.abs(bound[1] - bound[0]))
        total += exp_error
    return total / len(bounds)


bounds = calculate_all_bounds(N, X, Z, ZBAR, L_S)
print(calc_expected_error(bounds))
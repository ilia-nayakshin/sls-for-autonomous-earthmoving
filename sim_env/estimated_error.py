import numpy as np
import matplotlib.pyplot as plt

ZBAR = 100
DELTAZ = 2
L_S = 20
N = 40

X = 100
Z = 0


def surface(x):
    # surface is defined here.
    z = ZBAR
    # if x >= DELTAZ:
    #     z = ZBAR + DELTAZ
    # elif x <= -DELTAZ:
    #     z = ZBAR - DELTAZ
    # else:
    #     z = 100 + x
    return z


def calculate_bounds(n, x, z, n_pix, l_s):
    '''takes the position of the camera and the test surface and a given 
    pixel and calculates the theoretical error bounds for that pixel.'''
    x_centre = n_pix * (l_s / n)
    x_lower = x_centre - (l_s / (2 * n))
    x_upper = x_centre + (l_s / (2 * n))
    # calculate z of ends of the sub-line
    z_lower = surface(x_lower)
    z_upper = surface(x_upper)
    zbar = surface((x_lower + x_upper) / 2)
    # calculate gradient of the sub-line
    gradient = (z_upper - z_lower) / (x_upper - x_lower)
    # calculate maximum and minimum distance from the line
    a_1 = (z * (x - x_lower) + (zbar - z) * x) / ((x - x_lower) + (zbar - z) * (x_centre / zbar))
    a_2 = (z * (x - x_upper) + (zbar - z) * x) / ((x - x_upper) + (zbar - z) * (x_centre / zbar))
    # find scale to accomodate for non-zero gradient
    denominator = zbar - a_1
    numerator = zbar - (z_lower + gradient * ((a_1 * (x_centre / zbar)) - x_lower))
    scale = 1 - (numerator / denominator)
    # find error
    error_1, error_2 = (a_1 - zbar) * scale, (a_2 - zbar) * scale
    # print(scale, gradient, error_1, error_2)
    return [error_1, error_2]


def calculate_all_bounds(n, x, z, l_s):
    end = (n - 1) / 2
    if end == 0:
        pixels = [0]
    else:
        pixels = np.arange(-end, end + 1, 1)
    all_bounds = []
    # if odd, pix should be an integer. if odd, pix should be an integer +0.5.
    # centres, edges = calc_centres_and_edges(pixels, n, l_s)
    # print(centres)
    # print(' a')
    # print(edges)
    # for i in range(len(centres)):
        # all_bounds.append(calculate_bounds(n, x, z, edges[i], edges[i+1], centres[i]))
    for pix in pixels:
        all_bounds.append(calculate_bounds(n, x, z, pix, l_s))
    return all_bounds


# def calc_centres_and_edges(pixels, n, l_s):
#     delta_x = l_s / n
#     del_x = delta_x / 10000
#     centres = []
#     edges = []
#     for pix in pixels:
#         x_guess = pix * delta_x
#         # print(x_guess, end=' ')
#         x_guess = find_intersection(x_guess, del_x, ZBAR / x_guess)
#         # print(x_guess)
#         centres.append(x_guess)
#         left_x = (pix - 0.5) * delta_x
#         left_x = find_intersection(left_x, del_x, ZBAR / x_guess)
#         edges.append(left_x)
#     last_edge = (pixels[-1] + 0.5) * delta_x
#     edges.append(find_intersection(last_edge, delta_x, ZBAR / last_edge))
#     return centres, edges

def calc_expected_error(bounds):
    total = 0
    for bound in bounds:
        exp_error = (bound[0]**2 + bound[1]**2) / (2 * np.abs(bound[1] - bound[0]))
        if exp_error != np.inf:
            total += exp_error
    return total / len(bounds)

ns = np.arange(1, 100, 1)
for n in ns:
    print(calc_expected_error(calculate_all_bounds(n, X, Z, L_S)))


def heat_map(xx, zz, error):
    figure, axes = plt.subplots()
    error[error > 10**10] = 0
    c = plt.pcolormesh(xx, zz, error, norm='log', cmap='hot_r')
    # axes.set_title('Heatmap')
    axes.set_xlabel('X position')
    axes.set_ylabel('Z position')
    axes.axis([xx.min(), xx.max(), zz.min(), zz.max()])
    bar = figure.colorbar(c)
    bar.set_label("Estimated Error (%)")

    plt.show()


def generate_heat_map(n, l_s, x, z):
    xx, zz = np.meshgrid(x, z)
    error = np.zeros(np.shape(xx))
    for i_r, row in enumerate(xx):
        for i_c, x in enumerate(row):
            error[i_r, i_c] = calc_expected_error(calculate_all_bounds(n, x, zz[i_r, i_c], l_s))
    heat_map(xx, zz, error)



# def find_intersection(x_guess, delta_x, zgradient, tol=1e-6, iter=1000, learn_rate=0.00005):
#     grad = 0
#     max_hit = False
#     for i in range(iter):
#         z_guess = x_guess * zgradient
#         z_guess_abv = (x_guess + delta_x) * zgradient
#         z_guess_bel = (x_guess - delta_x) * zgradient

#         z_surf = surface(x_guess)
#         z_surf_abv = surface(x_guess + delta_x)
#         z_surf_bel = surface(x_guess - delta_x)

#         error = np.abs(z_surf - z_guess)
#         diff_abv = np.abs(z_surf_abv - z_guess_abv)
#         diff_bel = np.abs(z_surf_bel - z_guess_bel)

#         prev_grad = grad
#         grad = (diff_abv - diff_bel) / (2*delta_x)
#         if error < tol:
#             break
#         if i == iter - 1:
#             print("Max iterations hit without convergence. Adjust parameters.", z_surf, z_surf_abv, z_surf_bel, error)
#             max_hit = True
#         # print(x_guess, grad, error, z_surf, z_guess, z_surf_abv, z_guess_abv, z_surf_bel, z_guess_bel)
#         x_guess += - learn_rate * grad - learn_rate * 0.5 * prev_grad
#     if not max_hit:
#         print("max not hit.")
#     return x_guess
    

# x = np.arange(-100.03, 100.03, 0.1)
# z = np.arange(-50.03, 100.03, 1)
# generate_heat_map(N, L_S, x, z)

# bounds = calculate_all_bounds(N, X, Z, L_S)
# print(calc_expected_error(bounds))


# error = np.load("error_heatmap_n20_a0_slope1.0471975511965976.npz")['error']
# x = np.arange(-130, 140, 10)
# z = np.arange(-50, 100, 10)
# xx, zz = np.meshgrid(x, z)
# heat_map(xx, zz, error)

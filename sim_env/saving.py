import numpy as np
import cv2 as cv


def save_error_details(filename, sim_env, error, gvs):
    '''takes important details for making plots from variables passed in
    and saves them to a file with the appropriate labels.'''

    # setup key variables
    max_err = np.max(error)
    min_err = np.min(error)
    avg_err = np.average(error)
    proj_imgs = sim_env.projector.projected_imgs
    capt_imgs = sim_env.camera.captured_imgs
    t = gvs.T
    r = gvs.R
    k_proj = sim_env.projector.K
    k_cam = sim_env.camera.K
    n = gvs.PPIX_NUM
    a = (gvs.CPIX_NUM / n) - 1

    # save file
    np.savez(filename, error=error, max_err=max_err, min_err=min_err,
             avg_err=avg_err, proj_imgs=proj_imgs, capt_imgs=capt_imgs,
             t=t, r=r, k_proj=k_proj, k_cam=k_cam, n=n, a=a)


def load_error_details(filename):
    '''takes filename and loads the key details. returns all details, accessible
    by a dict'''
    file = np.load(filename)
    return file


def get_all_files(filenames):
    '''takes list of file names separated by commas. names should not include .npz.
    returns dicts of each file.'''
    files = filenames.split(',')
    all_files = []
    for file in files:
        name = (file + '.npz').replace('\n', '') # remove newlines
        all_files.append(load_error_details(name))
    return all_files


def get_mean_errors(files):
    '''takes array of file dicts and returns array of mean error for each file.'''
    mean_errors = []
    for file in files:
        error = file['error']
        total_error = np.nansum(error)
        count = np.count_nonzero(~np.isnan(error)) # counts only non-nan elements
        if count == 0 and total_error == 0: 
            mean_errors.append(np.nan)
        else: 
            mean_errors.append(total_error / count)
    return mean_errors


def get_std_errors(files, means):
    '''takes array of file dicts and mean errors and returns array of std of errors.'''
    std_errors = []
    for i, mean in enumerate(means):
        error = files[i]['error']
        count = np.count_nonzero(~np.isnan(error))
        if count == 0:
            std = np.nan
        else:
            error.flatten() # for following calculation
            error = error[~np.isnan(error)]
            std = np.sqrt((1 / count) * np.sum([(item - mean)**2 for item in error]))
        std_errors.append(std)
    return std_errors


def get_max_errors(files):
    '''returns maximum error in the given files.'''
    max_errors = []
    for file in files:
        error = file['error'].flatten()
        error = error[~np.isnan(error)]
        if len(error) > 0:
            max_errors.append(np.max(error))
        else:
            max_errors.append(np.nan)
    return max_errors


filenames = '''n-1_a-0_t-0-0-20,
n-1_a-1_t-0-0-20,
n-1_a-2_t-0-0-20,
n-2_a-0.5_t-0-0-20,
n-2_a-0_t-0-0-20,
n-2_a-1_t-0-0-20,
n-2_a-2_t-0-0-20,
n-3_a-0.4_t-0-0-20,
n-3_a-0.5_t-0-0-20,
n-3_a-0_t-0-0-20,
n-3_a-1_t-0-0-20,
n-3_a-2_t-0-0-20,
n-4_a-0_t-0-0-20,
n-5_a-0_t-0-0-20,
n-6_a-0_t-0-0-20,
n-7_a-0_t-0-0-20,
n-8_a-0_t-0-0-20,
n-9_a-0_t-0-0-20,
n-10_a-0_t-0-0-20,
n-11_a-0_t-0-0-20,
n-12_a-0_t-0-0-20,
n-14_a-0_t-0-0-20,
n-16_a-0_t-0-0-20,
n-18_a-0_t-0-0-20,
n-20_a-0.5_t-0-0-20,
n-20_a-0_t-0-0-20,
n-20_a-1_t-0-0-20'''



# files = get_all_files(filenames)
# mean_errors = get_mean_errors(files)
# std_errors = get_std_errors(files, mean_errors)
# max_errors = get_max_errors(files)
# for std in max_errors:
#     print(std)
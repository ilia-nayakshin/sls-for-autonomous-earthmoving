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
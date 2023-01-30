# External
import os

import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from multiprocessing import Pool
import xarray as xr
import cv2
# Common

# bat_ds = xr.load_dataset("/nexsan/people/xbxu/HYCOM/ATLc0.02/E026surf_STNA/depth_ATLc0.02_04_STNA.nc")
# bath = bat_ds.depth.data
inc_res_factor = 4
# THESE SHOULD BE THE SIZES OF THE LR FILES (this should be the same at demo.py line 82)
# crop_dims = [100, 100, 3] # Remember initial data is already 400x400
# crop_dims = [478, 1120, 3]  # For training and testing with trained data
# crop_dims = [1744, 1240, 3] # For testing 202105
crop_dims = [696, 1000, 3] # For new training data from /nexsan/people/xbxu/HYCOM/ATLc0.02/E026surf_DARPA
def generateX_202105(input_file):
    """
    Generates the X,Y from the input file
    :param input_file:
    :return:
    """

    # print(input_file)
    ds = xr.load_dataset(input_file)
    u = ds.surf_u[0,:,:]
    v = ds.surf_v[0,:,:]
    ds.close()

    return u[:crop_dims[0], :crop_dims[1]], v[:crop_dims[0], :crop_dims[1]]

def generateXYNewPred(input_file):
    """
    Generates the X,Y from the input file
    :param input_file:
    :return:
    """
    # print(input_file)
    ds = xr.load_dataset(input_file)
    u = ds.uvel[0,:,:]
    v = ds.vvel[0,:,:]
    ds.close()

    return u[:crop_dims[0], :crop_dims[1]], v[:crop_dims[0], :crop_dims[1]]

def generateXY(input_file):
    """
    Generates the X,Y from the input file
    :param input_file:
    :return:
    """
    # print(input_file)
    ds = xr.load_dataset(input_file)
    u = ds.uvel[0,:,:]
    v = ds.vvel[0,:,:]
    ds.close()

    u_red = cv2.resize(u.values, None, fx=1/inc_res_factor, fy=1/inc_res_factor)
    v_red = cv2.resize(v.values, None, fx=1/inc_res_factor, fy=1/inc_res_factor)
    uy = u.values[:crop_dims[0]*inc_res_factor, :crop_dims[1]*inc_res_factor]
    vy = v.values[:crop_dims[0]*inc_res_factor, :crop_dims[1]*inc_res_factor]

    # return u_red[:crop_dims[0], :crop_dims[1]], v_red[:crop_dims[0], :crop_dims[1]], uy, vy, bath[:crop_dims[0]*inc_res_factor, :crop_dims[1]*inc_res_factor]
    return u_red[:crop_dims[0], :crop_dims[1]], v_red[:crop_dims[0], :crop_dims[1]], uy, vy, 0

def mygenerator(idxs):
    # input_folder = "/data/DARPA/SuperResolution/Data"
    # input_folder = "/nexsan/people/xbxu/HYCOM/ATLc0.02/E026surf_STNA"
    input_folder = "/nexsan/people/xbxu/HYCOM/ATLc0.02/E026surf_DARPA"
    # IMPORTANT you also need to modify line 59 and 70 in data.py to match the number of examples
    curr_idx = -1 # First index to use
    file_names = os.listdir(input_folder)
    file_names.sort()
    # file_names = np.array([x for x in file_names if x.find("daily") != 0])
    file_names = np.array([x for x in file_names if x.find("026") != -1])
    file_names = file_names[idxs]
    np.random.shuffle(file_names) # We shuffle the folders every time we have tested all the examples

    # Obtain an ocean mask from the first input file
    _, domain, _, _, _ = generateXY(join(input_folder,file_names[0]))
    mask_idx = np.logical_not(np.isnan(domain)) # Mask with ones inside the ocean
    mask = np.zeros(domain.shape)
    mask[mask_idx] = 1

    X = np.zeros((crop_dims[0], crop_dims[1], 3), dtype=np.float16)
    Y = np.zeros((crop_dims[0]*inc_res_factor, crop_dims[1]*inc_res_factor, 3), dtype=np.float16)
    while True:
        # Position where there is a mask
        try:
            # *********************** Reading data **************************
            if curr_idx < (len(file_names) - 1):
                curr_idx += 1
            else:
                curr_idx = 0
                np.random.shuffle(file_names) # We shuffle the folders every time we have tested all the examples

            c_file = file_names[curr_idx]
            u_red, v_red, u, v, _= generateXY(join(input_folder,c_file))
            # print(F"After generateXY u_red:{u_red.shape} v_red:{v_red.shape} u:{u.shape}, v:{v.shape} mask:{mask.shape}")
            # print(F"X: {X.shape} Y: {Y.shape}")
            X[:,:,0] = u_red
            X[:,:,1] = v_red
            X[:,:,2] = mask

            Y[:,:,0] = u
            Y[:,:,1] = v
            Y[:,:,2] = 0

            np.nan_to_num(X,0)
            np.nan_to_num(Y,0)

            yield X, Y

        except Exception as e:
            print("----- Not able to generate for: ", curr_idx, " ERROR: ", str(e))
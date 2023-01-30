import imageio
import os
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import cv2
import cmocean
# Common
from img_viz.eoa_viz import EOAImageVisualizer

hr_folder = "/data/DARPA/SuperResolution/DataImgs/Validation/HR"
lr_folder = "/data/DARPA/SuperResolution/DataImgs/Validation/LR"
nn_folder = "/data/DARPA/SuperResolution/DataImgs/Predictions"
output_folder = "/data/DARPA/SuperResolution/DataImgs/Predictions_Diff"

hr_files = os.listdir(hr_folder)
lr_files = os.listdir(lr_folder)
nn_files = os.listdir(nn_folder)

for lr_file, hr_file, nn_file in zip(lr_files, hr_files, nn_files):
    hr_im = imageio.imread(join(hr_folder, hr_file)).astype(np.float32)
    lr_im = imageio.imread(join(lr_folder, lr_file)).astype(np.float32)
    nn_im = imageio.imread(join(nn_folder, nn_file)).astype(np.float32)
    hr_cub_im = cv2.resize(lr_im, None, fx=4, fy=4)

    u = hr_im[:,:,0]
    v = hr_im[:,:,1]
    u_cub = hr_cub_im[:,:,0]
    v_cub = hr_cub_im[:,:,1]
    u_nn = nn_im[:,:,0]
    v_nn = nn_im[:,:,1]

    u_cub_diff = u - u_cub
    v_cub_diff = v - v_cub
    u_nn_diff = u - u_nn
    v_nn_diff = v - v_nn
    nn_diff = hr_im - nn_im
    rmse_cub_u = np.mean(np.sqrt((u-u_cub)**2))
    rmse_cub_v = np.mean(np.sqrt((v-v_cub)**2))
    rmse_nn_u = np.mean(np.sqrt((u-u_nn)**2))
    rmse_nn_v = np.mean(np.sqrt((v-v_nn)**2))

    def_cmap = cmocean.cm.delta
    def_cmap_err = cmocean.cm.diff
    # ---------------- U -----------------
    minval = 0
    maxval = 255
    minval_diff = -10
    maxval_diff = 10
    dl = cmocean.cm.delta
    diff = cmocean.cm.delta
    img_viz_ob = EOAImageVisualizer(output_folder=output_folder, disp_images=False,
                                    mincbar=[minval, minval, minval, minval_diff, minval_diff,
                                             minval, minval, minval, minval_diff, minval_diff],
                                    maxcbar=[maxval, maxval, maxval, maxval_diff, maxval_diff,
                                             maxval, maxval, maxval, maxval_diff, maxval_diff])
    img_viz_ob.plot_2d_data_np_raw([u, u_cub, u_nn, u_cub_diff, u_nn_diff,
                                    v, v_cub, v_nn, v_cub_diff, v_nn_diff],
                                   var_names=['U HR', 'U Cubic', 'U NN', F'U Diff cubic rmse:{rmse_cub_u:0.2f}', F'U Diff NN rmse:{rmse_nn_u:0.2f}',
                                              'V HR', 'V Cubic', 'V NN', F'V Diff cubic rmse:{rmse_cub_v:0.2f}', F'V Diff NN rmse:{rmse_nn_v:0.2f}'],
                                   file_name=lr_file,
                                   cmap=[dl, dl, dl, diff, diff,
                                         dl, dl, dl, diff, diff],
                                   cols_per_row=5)


import os
import argparse

import tensorflow as tf
from tensorflow.keras.utils import plot_model

from scipy.interpolate import interp1d

import data
from model import get_generator
import utils

from Generators import generateXY, generateX_202105, generateXYNewPred
import cmocean
# from img_viz.eoa_viz import EOAImageVisualizer
import cv2
import numpy as np
import xarray as xr
from os.path import join


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# We need this file to copy the grid information
grd_file = "/nexsan/people/xbxu/HYCOM/ATLc0.02/E026surf_DARPA/026_archm.0020_348_12_DARPA_daily.nc"
hr_ds = xr.load_dataset(grd_file)

def save_image(image, save_dir, file_name, ext):
    image = (image * 127.5) + 127.5
    image = tf.cast(image, tf.uint8)
    image = tf.squeeze(image, axis=0)

    if ext == ".png":
        image = tf.image.encode_png(image)
    else:
        image = tf.image.encode_jpeg(image, quality=100, format='rgb')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    full_sr_path = os.path.join(save_dir, file_name + ext)
    tf.io.write_file(full_sr_path, image)
    print("Save a sr image at {}".format(full_sr_path))


def get_image(image_path, ext):
    image = data.load_and_preprocess_image(image_path, ext)
    image = tf.expand_dims(image, axis=0)
    return image


def sr_from_path(model, lr_path, save_dir):
    ext = utils.get_file_ext(lr_path)
    # out_dir = "/nexsan/people/ddmitry/DARPA/currents/202105_NNV2"
    out_dir = f"/nexsan/people/ddmitry/DARPA/currents/final_nn/{lr_path.split('/')[-2]}"
    if not(os.path.exists(out_dir)):
        os.makedirs(out_dir)
    name = lr_path.split("/")[-1]
    if os.path.exists(join(out_dir,name)):
        print(F"The file {name} already exists, skiping it!")
        return
    # lr_image = get_image(lr_path, ext)
    # Here I need to load the image and

    inc_res_factor = 4

    # ----------------- When we do know the HR Testing on Trained data ----------------
    # make_plot = True
    # save_netcdf = False
    # # crop_dims = [478, 1120, 3]  # For training (this should be the same at Generators.py)
    # crop_dims = [696, 1000, 3] # For new training data from /nexsan/people/xbxu/HYCOM/ATLc0.02/E026surf_DARPA
    # X = np.zeros((crop_dims[0], crop_dims[1], 3), dtype=np.float16)
    # u_red, v_red, u, v, bath = generateXY(lr_path)
    # hr_land = np.isnan(u) # Mask with ones inside the ocean
    # mask_idx = np.logical_not(np.isnan(u_red)) # Mask with ones inside the ocean
    # mask = np.zeros(u_red.shape)
    # mask[mask_idx] = 1
    # X[:,:,0] = u_red
    # X[:,:,1] = v_red
    # X[:,:,2] = mask
    # np.nan_to_num(X,0)
    # ----------------- END --------------

    # ----------------- FOR REAL PREDICITON NO HR IS KNOWN --------------
    make_plot = False
    save_netcdf = True
    # crop_dims = [1744, 1240, 3]
    crop_dims = [696, 1000, 3] # For new training data from /nexsan/people/xbxu/HYCOM/ATLc0.02/E026surf_DARPA
    X = np.zeros((crop_dims[0], crop_dims[1], 3), dtype=np.float16)
    u, v  = generateXYNewPred(lr_path)
    mask_idx = np.logical_not(np.isnan(u)) # Mask with ones inside the ocean
    mask = np.zeros(u.shape)
    mask[mask_idx] = 1
    X[:,:,0] = u
    X[:,:,1] = v
    X[:,:,2] = mask
    np.nan_to_num(X,0)
    # ----------------- END --------------

    # sr_image = model.predict(lr_image, steps=1)
    sr_image = model.predict(np.expand_dims(X,axis=0), steps=1)
    # sr_image = sr_image.clip(-1, 1)
    u_nn = sr_image[0,:,:,0]
    v_nn = sr_image[0,:,:,1]

    # ------------------------------------------
    if make_plot:
        print(F"After prediction {lr_path}")
        print(F"Saving at {save_dir}")
        u_nn[hr_land] = np.nan
        v_nn[hr_land] = np.nan
        rmse = np.nanmean(np.sqrt(((u-u_nn)**2 + (v-v_nn)**2)))
        inc_res_factor = 4
        u_cubic = cv2.resize(u_red, None, fx=inc_res_factor, fy=inc_res_factor)
        v_cubic  = cv2.resize(v_red, None, fx=inc_res_factor, fy=inc_res_factor)
        u_diff = u - u_cubic
        v_diff = v - v_cubic
        u_diff_nn = u_nn - u
        v_diff_nn = v_nn - v
        minval = -2
        maxval = 2
        st_lat = np.random.randint(0,200,1)[0]
        st_lon= np.random.randint(0,800,1)[0]
        red = [st_lat,st_lat+200, st_lon, st_lon+200]
        disp_images = True
        # img_viz_ob = EOAImageVisualizer(output_folder=F"{save_dir}Imgs/RCAN", disp_images=disp_images,
        #                                 mincbar=[minval, minval, minval, minval, minval,minval],
        #                                 maxcbar=[maxval, maxval, maxval, maxval, maxval, maxval])
        # img_viz_ob.plot_2d_data_np_raw([u_cubic[red[0]:red[1],red[2]:red[3]], u[red[0]:red[1],red[2]:red[3]], u_nn[red[0]:red[1],red[2]:red[3]],
        #                                 v_cubic[red[0]:red[1],red[2]:red[3]], v[red[0]:red[1],red[2]:red[3]], v_nn[red[0]:red[1],red[2]:red[3]]],
        #                                title=F"RMSE: {rmse:0.3f}",
        #                                var_names=[F'U1/12 {X.shape}', F'U1/50 {u.shape}', F'UNN {u_nn.shape}',
        #                                           F'V1/12 {X.shape}', F'V1/50 {v.shape}', F'VNN {v_nn.shape}'],
        #                                file_name=F"OUTPUT_{name}",
        #                                cmap=cmocean.cm.delta, cols_per_row=3)


        # minval = -.1
        # maxval = .1
        # img_viz_ob = EOAImageVisualizer(output_folder=F"{save_dir}Imgs/RCAN", disp_images=disp_images,
        #                                 mincbar=[minval, minval, minval, minval, minval,minval],
        #                                 maxcbar=[maxval, maxval, maxval, maxval, maxval, maxval])
        # img_viz_ob.plot_2d_data_np_raw([u_diff[red[0]:red[1],red[2]:red[3]],  u_diff_nn[red[0]:red[1],red[2]:red[3]],
        #                                 v_diff[red[0]:red[1],red[2]:red[3]],  v_diff_nn[red[0]:red[1],red[2]:red[3]]],
        #                                title=F"RMSE: {rmse:0.3f}",
        #                                var_names=[F'U1/12 MAE {np.nanmean(np.abs(u_diff)):0.3f}', F'UNN MAE {np.nanmean(np.abs(u_diff_nn)):0.3f}',
        #                                           F'V1/12 MAE {np.nanmean(np.abs(v_diff)):0.3f}', F'VNN MAE {np.nanmean(np.abs(v_diff_nn)):0.3f}'],
        #                                file_name=F"DIFF_{name}",
        #                                cmap=cmocean.cm.diff, cols_per_row=2)

    if save_netcdf:
        ds = xr.load_dataset(lr_path)
        out_dims = (crop_dims[0]*inc_res_factor, crop_dims[1]*inc_res_factor)
        # new_lat = np.linspace(ds.latitude[0], ds.latitude[crop_dims[0]], out_dims[0])
        # new_lon = np.linspace(ds.longitude[0], ds.longitude[crop_dims[1]], out_dims[1])
        curvilinear = True # Curvilinear grid
        if curvilinear:
            hr_grid = xr.load_dataset("/nexsan/people/xbxu/HYCOM/ATLc0.02/E026surf_DARPA/026_archm.0020_348_12_DARPA_daily.nc")
            # -------------------- New version
            nds = xr.Dataset(
                {
                    "uvel": (("time", "j-index","i-index"), np.expand_dims(u_nn, axis=0)),
                    "vvel": (("time", "j-index","i-index"), np.expand_dims(v_nn, axis=0)),
                },
                {"time": ds.time.data,
                 })
            nds.to_netcdf(join(out_dir,name), mode='w')

            grd_file = "grd_DARPA_GOFS3.1.nc"
            if not(os.path.exists(out_dir)):
                os.system(F"cp {grd_file} {out_dir}")
        else:

            # -------- for regular grid Interpolated lat and lon (1D) ----
            flat = interp1d(np.linspace(0,1,crop_dims[0]), ds.latitude[0:crop_dims[0]])
            flon = interp1d(np.linspace(0,1,crop_dims[1]), ds.longitude[0:crop_dims[1]])
            new_lat = flat(np.linspace(0,1,out_dims[0]))
            new_lon = flon(np.linspace(0,1,out_dims[1]))

            nds = xr.Dataset(
                {
                    "surf_u": (("time", "lat","lon"), np.expand_dims(u_nn, axis=0)),
                    "surf_v": (("time", "lat","lon"), np.expand_dims(v_nn, axis=0)),
                    "latitude": (("lat"), new_lat),
                    "longitude": (("lon"), new_lon),
                },
                {"time": ds.time.data, "lat":np.arange(out_dims[0]), "lon":np.arange(out_dims[1])})
            nds.to_netcdf(join(out_dir,name), mode='w')

def sr_from_folder(model, lr_dir, save_dir, ext):
    if lr_dir is not None:
        if not os.path.exists(lr_dir):
            raise Exception('Not found folder: ' + lr_dir)
        lr_paths = utils.get_image_paths(lr_dir, ext)
        lr_paths.sort()
        for lr_path in lr_paths:
            if (lr_path.find("grd") != - 1) or (lr_path.find("navgem") != - 1):
                continue
            sr_from_path(model, lr_path, save_dir)


def main():
    parser = argparse.ArgumentParser(description='Generate SR images')
    parser.add_argument('--arc', required=True, type=str, help='Model architecture')
    parser.add_argument('--model_path', required=True, type=str, help='Path to a model')
    parser.add_argument('--lr_dir', type=str, default=None, help='Path to lr images')
    parser.add_argument('--lr_path', type=str, default=None, help='Path to a lr image')
    parser.add_argument('--ext', type=str, help='Image extension')
    parser.add_argument('--default', action='store_true', help='Path to lr images')
    parser.add_argument('--save_dir', type=str, help='folder to save SR images')
    parser.add_argument('--cuda', type=str, default=None, help='a list of gpus')
    args = parser.parse_args()

    if args.cuda is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    # global sess
    model = get_generator(args.arc, is_train=False)
    print("** Loading model at: " + args.model_path)
    model.load_weights(args.model_path)
    plot_model(model, to_file="/home/olmozavala/Delete/super_resolution_old_delete/model.png", show_shapes=True, dpi=600)

    if args.default:
        lr_dirs = [os.path.join("./data/test/", dataset, "LR") for dataset in ["Set5", "Set14", "BSDS100"]]
        save_dirs = [os.path.join("./output/", args.arc, dataset) for dataset in ["Set5", "Set14", "BSDS100"]]
        for lr_dir, save_dir in zip(lr_dirs, save_dirs):
            sr_from_folder(model, lr_dir, save_dir, ".png")
    else:
        sr_from_folder(model, args.lr_dir, args.save_dir, args.ext)
        if args.lr_path is not None:
            sr_from_path(model, args.lr_path, args.save_dir)


if __name__ == '__main__':
    main()

#  OZ params
# Real training
# --arc=rcan --lr_dir=/nexsan/people/xbxu/HYCOM/ATLc0.02/E026surf_STNA --ext=.nc --save_dir=/data/DARPA/SuperResolution/Predictions --model_path=/home/olmozavala/Delete/super_resolution_old_delete/exp/rcan-11-01-14:03_Respaldo/cp-0002.h5 --cuda=0
# --arc=rcan --lr_dir=/nexsan/people/xbxu/HYCOM/ATLc0.02/E026surf_STNA --ext=.nc --save_dir=/data/DARPA/SuperResolution/Predictions --model_path=/home/olmozavala/Dropbox/MyProjects/EOAS/COAPS/Keras-Image-Super-Resolution/exp/rcan-10-26-11:25/cp-0002.h5 --cuda=0
# --arc=rcan --lr_dir=/nexsan/people/xbxu/HYCOM/ATLc0.02/E026surf_STNA --ext=.nc --save_dir=/data/DARPA/SuperResolution/Predictions --model_path=/home/olmozavala/Dropbox/MyProjects/EOAS/COAPS/Keras-Image-Super-Resolution/exp/rcan-10-26-11:25/cp-0002.h5 --cuda=0
# Cropped one
# --arc=rcan --lr_dir=/data/DARPA/SuperResolution/Data --ext=.nc --save_dir=/data/DARPA/SuperResolution/DataImgs/Predictions --model_path=/home/olmozavala/Dropbox/MyProjects/EOAS/COAPS/Keras-Image-Super-Resolution/exp/rcan-10-25-15:10/cp-0014.h5 --cuda=0
# --arc=rcan --lr_dir=/data/DARPA/SuperResolution/DataImgs/Validation/LR --ext=.png --save_dir=/data/DARPA/SuperResolution/DataImgs/Predictions --model_path=/home/olmozavala/Dropbox/MyProjects/EOAS/COAPS/Keras-Image-Super-Resolution/exp/rcan-10-21-17:04/final_model.h5 --cuda=0

# python demo.py --default --arc=erca --model_path=exp/erca-06-24-21\:12/final_model.h5 --cuda=0
# python demo.py --arc=erca --lr_path=../SRFeat/data/test/Set5/LR/head.png --save_dir=./output/Set5 --model_path=exp/erca-06-24-21\:12/final_model.h5 --cuda=0
# python demo.py --arc=erca --lr_dir=../SRFeat/data/test/Set5/LR --ext=.png --save_dir=./output/Set5 --model_path=exp/erca-06-24-21\:12/final_model.h5 --cuda=0
import xarray as xr
import matplotlib.pyplot as plt
import cmocean
import cv2
import cartopy.crs as ccrs
import numpy as np
from img_viz.eoa_viz import EOAImageVisualizer
from scipy.interpolate import interp2d

##

# lr_file = "/nexsan/people/ddmitry/DARPA/currents/202105/hycom_glby_930_2021052912_t000_uv3z.nc"
# hr_file = "/nexsan/people/ddmitry/DARPA/currents/202105_NN/30_2021052912_t000_uv3z.nc"
# hr_file = "/nexsan/people/xbxu/HYCOM/ATLc0.02/E026surf_DARPA/026_archm.0020_315_12_DARPA_daily.nc"
hr_file = "/nexsan/people/xbxu/HYCOM/ATLc0.02/E026surf_STNA/026_archm.0016_122_12_STNA_daily.nc"
hr_nn_file = ""

# hr_ds = xr.load_dataset(hr_nn_file)
hr_ds = xr.load_dataset(hr_file)

def plotDataset(ds, extent, title, file_name):
    print("Plotting...")
    lats = ds.latitude
    lons = ds.longitude

    # extent = (lons.min() - 360, lons.max() - 360, lats.min(), lats.max())
    extent = (lons.min(), lons.max(), lats.min(), lats.max())
    img_extent = (-180, 180, -90, 90)

    fig, ax = plt.subplots(1, 1, figsize=(12,5), subplot_kw={'projection': ccrs.PlateCarree()})
    # Left plot
    img = plt.imread('/home/olmozavala/Dropbox/TutorialsByMe/Python/PythonExamples/Python/MatplotlibEx/map_backgrounds/bluemarble_5400x2700.jpg')
    # ax.imshow(img, origin='upper', extent=img_extent, transform=ccrs.PlateCarree())
    # ax.imshow(ds.uvel.data.squeeze(), origin='lower', extent=extent, transform=ccrs.PlateCarree(), cmap=cmocean.cm.balance)
    ax.contourf(lons, lats, ds.uvel.data.squeeze(), 256, cmap=cmocean.cm.balance, extent=extent)
    ax.set_title(title)
    ax.coastlines()
    gl = ax.gridlines(draw_labels=True, color='grey', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.left_labels = False
    plt.tight_layout()
    plt.savefig(F'./figs/{file_name}')
    # plt.show()
    plt.close()
    print("Done!")

def plotNPArray(np, extent, title, lats, lons, file_name, figsize, cmap=cmocean.cm.balance):
    print("Plotting...")

    extent = (lons.min(), lons.max(), lats.min(), lats.max())
    img_extent = (-180, 180, -90, 90)

    fig, ax = plt.subplots(1, 1, figsize=figsize, subplot_kw={'projection': ccrs.PlateCarree()})
    # Left plot
    img = plt.imread('/home/olmozavala/Dropbox/TutorialsByMe/Python/PythonExamples/Python/MatplotlibEx/map_backgrounds/bluemarble_5400x2700.jpg')
    # ax.imshow(img, origin='upper', extent=img_extent, transform=ccrs.PlateCarree())
    im = ax.contourf(lons, lats, np, 256, cmap=cmap, extent=extent)
    ax.set_title(title)
    ax.coastlines()
    gl = ax.gridlines(draw_labels=True, color='grey', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.left_labels = False
    plt.tight_layout()
    plt.colorbar(im, location='right', shrink=.6, pad=.08)
    plt.savefig(F'./figs/{file_name}')
    # plt.show()
    plt.close()
    print("Done!")

##
dims_hr = hr_ds.uvel.shape
temp = np.zeros((dims_hr[1], dims_hr[2], 3))
temp[:,:,0] = hr_ds.uvel.squeeze()
lr_np = cv2.resize(temp, None, fx=1/4, fy=1/4)[:dims_hr[1],:dims_hr[2], 0]

## Extent
# extent = [260, 359, 0, 70]  # Whole domain
# extent = [300-360, 310-360, 40,45]  # Not sure LC
extent = [-88.11, -79, 22.1, 30]  # Florida
# extent = [290, 320, 30,55]  # North west part (Quebec latitude)
crop_hr_ds = hr_ds.where((hr_ds.latitude >= extent[2]) & (hr_ds.latitude <= extent[3]) &
                         (hr_ds.longitude >= extent[0]) & (hr_ds.longitude <= extent[1]) , drop=True)

# We need the indexes after
lat_hr_idxs =(hr_ds.latitude >= extent[2]) & (hr_ds.latitude <= extent[3])
lon_hr_idxs =(hr_ds.longitude >= extent[0]) & (hr_ds.longitude <= extent[1])
# crop_hr_ds = hr_ds.where(lat_hr_idxs & lon_hr_idxs, drop=True)
lats_crop = hr_ds.latitude[lat_hr_idxs]
lons_crop = hr_ds.longitude[lon_hr_idxs]
crop_hr_np = hr_ds.uvel[0,lat_hr_idxs,lon_hr_idxs]
temp = np.zeros((len(lats_crop), len(lons_crop), 3))
temp[:,:,0] = crop_hr_np
crop_lr_np_rgb = cv2.resize(temp, None, fx=1/4, fy=1/4)
crop_lr_np =  crop_lr_np_rgb[:,:,0]
crop_hr_bi_np = cv2.resize(crop_lr_np_rgb, None, fx=4, fy=4)[:,:,0]
# print(crop_hr_np.shape)
# print(crop_lr_np_rgb.shape)
# print(crop_hr_bi_np.shape)

## Plot HR and LR whole domain
# plotDataset(hr_ds, extent, 'HR 1/50', 'hr_whole_domain.png')
# lats = hr_ds.latitude
# lons = hr_ds.longitude
# plotNPArray(lr_np, extent, 'LR 1/25 ', lats[2::4], lons[1::4], 'lr_whole_domain.png', (12,5))

## PLot HR, LR and diff cropped domain
plotNPArray(crop_hr_np, extent, 'HR 1/50', lats_crop, lons_crop, 'hr_crop_domain.png', (11,8))
plotNPArray(crop_lr_np, extent, 'LR 1/25 ', lats_crop[1::4], lons_crop[::4], 'lr_crop_domain.png', (11,8))

##  Compute difference between hr and bi-cubic interpolation
diff = crop_hr_bi_np-crop_hr_np
plotNPArray(diff, extent, 'Diff 1/50', lats_crop, lons_crop, 'diff_hr_crop_domain.png', (11,8), cmap=cmocean.cm.diff)

##

crop_diff = diff.where(lat_hr_idxs & lon_hr_idxs, drop=True)

lats = crop_hr_ds.latitude
lons = crop_hr_ds.longitude

extent = (lons.min() - 360, lons.max() - 360, lats.min(), lats.max())
# extent = (lons.min(), lons.max(), lats.min(), lats.max())
fig, ax = plt.subplots(1, 1, figsize=(14,10), subplot_kw={'projection': ccrs.PlateCarree()})
# Left plot
img = plt.imread('/home/olmozavala/Dropbox/TutorialsByMe/Python/PythonExamples/Python/MatplotlibEx/map_backgrounds/bluemarble_5400x2700.jpg')
img_extent = (-180, 180, -90, 90)
ax.imshow(img, origin='upper', extent=img_extent, transform=ccrs.PlateCarree())
ax.imshow(crop_diff, origin='lower', extent=extent,
          transform=ccrs.PlateCarree(), cmap=cmocean.cm.balance)
# ax.coastlines()
gl = ax.gridlines(draw_labels=True, color='grey', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.left_labels = False
plt.tight_layout()
# plt.savefig('./figs/domain.png')
plt.show()
plt.close()

##  Show Difference between hr and synthetic hr
lats = crop_hr_ds.latitude
lons = crop_hr_ds.longitude

extent = (lons.min() - 360, lons.max() - 360, lats.min(), lats.max())
# extent = (lons.min(), lons.max(), lats.min(), lats.max())
fig, ax = plt.subplots(1, 1, figsize=(14,10), subplot_kw={'projection': ccrs.PlateCarree()})
# Left plot
img = plt.imread('/home/olmozavala/Dropbox/TutorialsByMe/Python/PythonExamples/Python/MatplotlibEx/map_backgrounds/bluemarble_5400x2700.jpg')
img_extent = (-180, 180, -90, 90)
# ax.imshow(img, origin='upper', extent=img_extent, transform=ccrs.PlateCarree())
ax.imshow(crop_lr_ds.uvel.data[0,:,:], origin='lower', extent=extent,
          transform=ccrs.PlateCarree(), cmap=cmocean.cm.balance)
ax.coastlines()
gl = ax.gridlines(draw_labels=True, color='grey', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.left_labels = False
plt.tight_layout()
# plt.savefig('./figs/domain.png')
plt.show()
plt.close()
fig, axs = plt.subplots(1, 3, figsize=(15,5), subplot_kw={'projection': ccrs.PlateCarree()})
axs[0].stock_img()
axs[0].set_title("LR")
im1 = axs[0].imshow(hr_ds.uvel.squeeze()[lr_lat_idxs,lr_lon_idxs], cmap=cmocean.cm.thermal, extent=[-1,1,-1,1])
axs[1].stock_img()
axs[1].set_title("HR NN")
im2 = axs[1].imshow(hr_ds.uvel.squeeze()[hr_lat_idxs,hr_lon_idxs], cmap=cmocean.cm.thermal, extent=[-1,1,-1,1])
axs[2].set_title("Diff")
axs[2].stock_img()
im3 = axs[2].imshow(diff, cmap=cmocean.cm.delta)
fig.colorbar(im1, ax=axs[0], shrink=0.7)
fig.colorbar(im2, ax=axs[1], shrink=0.7)
fig.colorbar(im3, ax=axs[2], shrink=0.7)
plt.show()

##

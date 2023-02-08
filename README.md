# Single Image Super Resolution, EDSR, SRGAN, SRFeat, RCAN, ESRGAN and ERCA (ours) benchmark comparison

# Code
## pretrain.py
This code is used to train a new model. It uses `png` images previoulsy generated. 
You can access these training images in wolf at `wolf.coaps.fsu.edu/data/DARPA/SuperResolution/DataImgs/`.
You can copy those images to skynet or run it direclty from wolf. 

Example use for training a new model:
`--arc=rcan --train=/data/DARPA/SuperResolution/DataImgs/Train --train-ext=.png --valid=/data/DARPA/SuperResolution/DataImgs/Validation --valid-ext=.png --cuda=1`

## demo.py
This code is used for testing. I still need to modify it to the original training data. 



# Paths

Input U and V for the Atlantic (netcdf):
`/nexsan/people/xbxu/HYCOM/ATLc0.02/E026surf_DARPA/`

Ouput `wolf.coaps.fsu.edu/data/DARPA/SuperResolution/`


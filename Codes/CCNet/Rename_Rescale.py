import os
# import io

import skimage.io as io
import cv2
import numpy as np
from matplotlib import pyplot as plt
 
# The folders of the original frames
img_gt_folder='./data/test/image'
img_seg_folder='./data/Seg_result1'
img_new_folder='./data/Original_frame'

# To get the original name

for i in range(0,21,1):
    gt_name = "frankurt_000000_00" + str(i).zfill(6)+ "_leftImg8bit.png"
    path_gt = img_gt_folder + "/"+ gt_name
    img_gt = cv2.imread(path_gt)
    # To save the frames
    io.imsave('./data/Original_frame/img' + str(step) + '.png', img_gt)
    
    seg_name = "seg" + str(i) + ".png"
    path_seg = img_seg_folder + "/" + seg_name
    img_seg = cv2.imread(path_seg, 0)
    img_seg = transform.rescale(img_seg, 2)
    io.imsave('./data/Seg_result/seg' + str(step) + '.png', img_seg)

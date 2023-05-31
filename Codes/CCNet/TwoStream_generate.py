import os
# import io

import skimage.io as io
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

# print a matrix where the '0'='1','1'='0'
def get_array_inverse(arr1):
#     arr= arr1
    arr= np.array(arr1)
    for i in range(0,arr.shape[0]):
        for j in range(0,arr.shape[1]):
            if(arr1[i][j] == 0):
                arr[i][j]= 1
            else:
                arr[i][j]=0
    return arr


# To get the mask for the uimportant and important area
class img_split():

    def __init__(self):
        super(img_split, self).__init__()

    #     # print a mask for the uimportant area
    #     def get_array_uim(arr1):
    #         arr=arr1
    #         for i in range(0,arr1.shape[0]):
    #             for j in range(0,arr1.shape[1]):
    #                 if(arr[i][j] == 0):
    #                     arr[i][j]= 0
    #                 else:
    #                     arr[i][j]=1
    #         return arr

    # print a mask for the important area
    def get_array_im(arr1):
        #         arr2 = arr1
        arr2 = np.array(arr1)
        row = arr2.shape[0]
        col = arr2.shape[1]
        for i in range(0, row):
            for j in range(0, col):
                if (arr1[i][j] == 0):
                    arr2[i][j] = 1
                else:
                    arr2[i][j] = 0
        return arr2

    # To get an 16*16 macroblock-based improved mask for the uimportant area
    def seg_mask_im_macro(arr2):
        #         arr1=arr2
        arr1 = np.array(arr2)
        row = arr1.shape[0]
        col = arr1.shape[1]
        nrow = int(row / 16)
        ncol = int(col / 16)
        for i in range(nrow):
            for j in range(ncol):
                slice1 = arr1[16 * i:16 * (i + 1), 16 * j:16 * (j + 1)]
                for i1 in range(16):
                    for j1 in range(16):
                        if (slice1[i1][j1] == 1):
                            arr1[16 * i:16 * (i + 1), 16 * j:16 * (j + 1)] = 1
                            break
                        break
        return arr1


# To change the color channel in images for saving
# Since the color is "B,G,R" in the opencv setting
def color_adjust(img):
    B,G,R = cv2.split(img)
    img=cv2.merge([R,G,B])

#######################################################################################

# The folders of the frames
img_gt_folder='./data/Original_frame'
img_seg_folder='./data/Seg_result'

# To get the background and the important images
def get_img(img_seg,img_gt):
    arr_im = img_split.get_array_im(img_seg)
    seg_mask_im_macro_old = img_split.seg_mask_im_macro(arr_im)
    # To get the mask into a 3 dimension array
    im = np.array(seg_mask_im_macro_old)
    #  to add a dimension to the "2" dimension of the arr
    #  Repeat represents the number of repeats
    # Axis represents the dimension in which the repeats are performed.
    seg_mask_im_macro = np.expand_dims(im,2).repeat(3,axis=2)
    # seg_mask_im_macro.shape
    # plt.imshow(seg_mask_im_macro)
    # seg_mask_uim_macro = get_array_inverse(seg_mask_im_macro)
    # plt.imshow(seg_mask_im_macro)
    im_images = seg_mask_im_macro * img_gt
    uim = get_array_inverse(im)
    seg_mask_uim_macro = np.expand_dims(uim,2).repeat(3,axis=2)
    uim_images = seg_mask_uim_macro * img_gt
    return im_images,  uim_images

###########################################################################################

# To save the frames
def uim_to_local_img(img, step):
    io.imsave('./data/Stream/background/BA' + str(step) + '.png', img)


def im_to_local_img(img, step):
    io.imsave('./data/Stream/important/IA' + str(step) + '.png', img)


# To iterat through the seg folder and

num_frame = 2197
time_start = time.time()
for i in range(0, 2, 1):
    gt_name = "img" + str(i) + ".png"
    seg_name = "seg" + str(i) + ".png"

    path_gt = img_gt_folder + "/" + gt_name
    path_seg = img_seg_folder + "/" + seg_name
    img_gt = cv2.imread(path_gt, 1)
    img_seg = cv2.imread(path_seg, 0)
    img_im, img_uim = get_img(img_seg, img_gt)
    color_adjust(img_im)
    color_adjust(img_uim)

    im_to_local_img(img_im, i)

    # img_uim = get_img(img_seg,img_gt)[1]
    uim_to_local_img(img_uim, i)

time_end = time.time()
time_cost = time_end - time_start

# To print the time in generating one frame
print('Average time cost per frame', time_cost / num_frame)

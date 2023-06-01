
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import collections
import os
import math

img_mask = cv2.imread("/home/wang3_y@WMGDS.WMG.WARWICK.AC.UK/Desktop/EfficientPS/output/6_GT.png",cv2.IMREAD_GRAYSCALE)
img_gt = cv2.imread("/home/wang3_y@WMGDS.WMG.WARWICK.AC.UK/code/deeplabv2/dataset/KITTI_STEP/images/train/0000/000000.png")
img_pre = cv2.imread("/home/wang3_y@WMGDS.WMG.WARWICK.AC.UK/Desktop/EfficientPS/output/5.png",cv2.IMREAD_GRAYSCALE)
im = cv2.imread("/home/wang3_y@WMGDS.WMG.WARWICK.AC.UK/Desktop/EfficientPS/mask-try.png")

diff = cv2.imread("/home/wang3_y@WMGDS.WMG.WARWICK.AC.UK/Desktop/EfficientPS/diff.png")
print(img_pre.shape)
# diff = img_gt - im

# cv2.imwrite("diff.png", diff)


# copy =  cv2.imread("/home/wang3_y@WMGDS.WMG.WARWICK.AC.UK/Desktop/EfficientPS/try2.png")

#  # print a mask for the important area

# print a mask for the important area
# def get_array_im(arr1):
#     #         arr2 = arr1
#     arr2 = np.array(arr1)
#     row = arr2.shape[0]
#     col = arr2.shape[1]
#     for i in range(0, row):
#         for j in range(0, col):
#             if (arr1[i][j] == 0):
#                 arr2[i][j] = 1
#             else:
#                 arr2[i][j] = 0
#     return arr2

# img6 = get_array_im(img_mask)
# print(img6)

# # img6=get_array_inverse(img_mask)
# # img6 = np.array(img6)
# # print(np.shape(img6))

# # cv2.imwrite("inverse.png",np.array(img6))

# # cv2.imwrite("try2.png",img_gt)
# # img0 = arr2 * img_gt
# # # img0 = np.expand_dims(arr2,2).repeat(3,axis=2) * img_gt
# # # cv2.imwrite("try6.png",arr2)

import os

import cv2

import numpy as np

# img_gt = "/home/wang3_y@WMGDS.WMG.WARWICK.AC.UK/code/deeplabv2/dataset/KITTI_STEP/images/train/0000/000000.png"
# img_pre = "/home/wang3_y@WMGDS.WMG.WARWICK.AC.UK/Desktop/EfficientPS/output/5.png"

# def add_mask2image_binary(images_path, masks_path, masked_path):
# # Add binary masks to images
#     img = cv2.imread(images_path)
#     # mask = cv2.imread(masks_path, cv2.IMREAD_GRAYSCALE)  # 将彩色mask以二值图像形式读取
#     mask = cv2.imread(masks_path)  # 将彩色mask以二值图像形式读取
#     # cv2.imwrite('/home/haonan_zhao/code/imagecut/mask/gray_mask2.png',mask)
#     masked = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask)  #将image的相素值和mask像素值相加得到结果
#     cv2.imwrite(masked_path, masked)

# images_path = img_gt
# masks_path = img_pre
# masked_path = 'mask-pre.png'
# add_mask2image_binary(images_path, masks_path, masked_path)

def get_array_im(arr1):
    #         arr2 = arr1
    # print(arr1[0][0])
    arr2 = np.array(arr1)
    row = arr2.shape[0]
    col = arr2.shape[1]
    for i in range(0, row):
        for j in range(0, col):
            if (arr1[i][j] == 0):
                arr2[i][j] = 0
            else:
                arr2[i][j] = 1
    return arr2

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
    print("trying arr1...")
    print(arr1)
    return arr1
 

img3 =get_array_im(img_pre)
# print(img3)
print("trying...")
# cv2.imwrite("0.png",img3 )
# img3 = np.array(seg_mask_im_macro(img3))
img3 = cv2.resize(img3, (1242,375))
img3 = np.expand_dims(img3,2).repeat(3,axis=2)
img3 = img3 * img_gt
cv2.imwrite("IM-pre.png",img3)

# def seg_mask_im_macro1(arr2):
#     #         arr1=arr2
#     arr1 = np.array(arr2)
#     row = arr1.shape[0]
#     col = arr1.shape[1]
#     nrow = int(row / 16)
#     ncol = int(col / 16)
#     for i in range(nrow):
#         for j in range(ncol):
#             slice1 = arr1[16 * i:16 * (i + 1), 16 * j:16 * (j + 1)]
#             for i1 in range(16):
#                 for j1 in range(16):
#                     if (slice1[i1][j1] == 1):
#                         arr1[16 * i:16 * (i + 1), 16 * j:16 * (j + 1)] = 255
#                         break
#                     break
#     print("trying arr1...")
#     print(arr1)
#     return arr1

# img4 =get_array_im(img_mask)

# img4 = np.array(seg_mask_im_macro1(img4))
# img4= np.expand_dims(img4,2).repeat(3,axis=2)
# img3 = cv2.resize(img3, (1242,375))
# cv2.imwrite("binary-mask.png",img4)
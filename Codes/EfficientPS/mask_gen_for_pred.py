# This is a sample Python script.
 
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
#from filecmp import cmp
 
import cv2
import operator
import copy


import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import collections
import os
import math
 
import numpy as np
 
def img_input(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # 使用 cv2 来打开图片
    return img
 
def trans(img):#转换颜色
    img2 = copy.deepcopy(img)
    Size = img.shape
    width = Size[0]
    length = Size[1]
    num = Size[2]
    list1_black = [0, 0, 0]
    uioa1 = [0,0,23]
    uioa2 = [0,0,21]
    uioa3 = [0,0,11]
    # list2_green = [0,128,0]
    color_lst=[]
    random_color_lst=[]
    for i in range(0,width):
        for j in range(0,length):
            # if operator.eq(img[i][j].tolist(),list1_black) == False:
            if operator.eq(img[i][j].tolist(),uioa1) == True or operator.eq(img[i][j].tolist(),uioa2) == True or operator.eq(img[i][j].tolist(),uioa3) == True:
                img2[i][j] = np.array([0,0,0])
            else:
                img2[i][j] = np.array([255,255,255])

    return img2

def get_array_im(arr1):
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
                        arr1[16 * i:16 * (i + 1), 16 * j:16 * (j + 1)] = 255
                        break
                    break
    # print("trying arr1...")
    # print(arr1)
    return arr1

# To generate the binary mask for a directionary for the prediction results
def Recolor_for_results(in_path, output_path):
    f1 = os.listdir(in_path)
    f1.sort(key=lambda x: int(x.split('.')[0]))
    total_num = len(f1)
    for i in range(0, total_num):
        img_path = in_path + f1[i]
        img = img_input(img_path)
        img = trans(img)
        cv2.imwrite(filename = output_path + str(i) + ".png",img = img)
        print("Transforming the {} th image color ...".format(i))

# def binary_mask_generation(in_path,out_path):
#     f1 = os.listdir(in_path)
#     f1.sort(key=lambda x: int(x.split('.')[0]))
#     total_num = len(f1)
#     for i in range(0, total_num):
#         img_pred_mask = cv2.imread(in_path +  f1[i], cv2.IMREAD_GRAYSCALE)
#         # print(img_pred_masks)
#         img = get_array_im(img_pred_mask)
#         img = np.array(seg_mask_im_macro(img))
#         img2 = cv2.resize(img, (1242,375))
#         cv2.imwrite(filename = out_path +  f1[i],img = img2)
#         print("Generating the %s th image macroblock mask..." %i)

# Recolor_for_results("/home/wang3_y@WMGDS.WMG.WARWICK.AC.UK/Desktop/EfficientPS/output/panopticmap/panopticmap/", 
# "/home/wang3_y@WMGDS.WMG.WARWICK.AC.UK/Desktop/EfficientPS/output/binary-mask/")
# print("..............Recolor done!.............")

in_path = "/home/wang3_y@WMGDS.WMG.WARWICK.AC.UK/Desktop/EfficientPS/output_265_28/output_265_28/"
out_path = "/home/wang3_y@WMGDS.WMG.WARWICK.AC.UK/Desktop/EfficientPS/output_265_28/recolor/"
Recolor_for_results(in_path,out_path)

# img3 = cv2.resize(img3, (1242,375))
# img3 = np.expand_dims(img3,2).repeat(3,axis=2)
# img3 = img3 * img_gt
# cv2.imwrite("IM-pre.png",img3)

 
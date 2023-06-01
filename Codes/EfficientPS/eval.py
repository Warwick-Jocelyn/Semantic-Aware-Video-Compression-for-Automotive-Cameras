
import numpy as np
from PIL import Image
from os.path import join

import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import collections
import os
import math

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    # print(k)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)
    # print(np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n))

def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1)
    # print(np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1))

def per_class_PA(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1)

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

def compute_mIoU(gt_dir, pred_dir,num_classes, name_classes):
    print('Num classes', num_classes)

    hist = np.zeros((num_classes, num_classes))
    gt_list = os.listdir(gt_dir)
    gt_list.sort(key=lambda x: int(x.split('.')[0]))
    pred_list = os.listdir(pred_dir)
    pred_list.sort(key=lambda x: int(x.split('.')[0]))
    gt_imgs = [join(gt_dir, x ) for x in gt_list]
    # print(len(gt_imgs))
    pred_imgs = [join(pred_dir, x) for x in pred_list[:154]]
    total_num = len(gt_imgs)

    # for ind in range(10):
    for ind in range(total_num):
        img = Image.open(pred_imgs[ind]).convert("L")
        img = img.resize((1242,375))
        pred = np.array(get_array_im(np.array(img)))
        # print(pred,pred.shape)
        img1 = Image.open(gt_imgs[ind]).convert("L")
        img1 = img1.resize((1242,375))
        label = np.array(get_array_im(np.array(img1)))
        # print(label,label.shape)
        # pred = np.array(get_array_im(Image.open(pred_imgs[ind]))
        # label = np.array(get_array_im(cv2.imread(gt_imgs[ind])))
        # label = np.array(Image.open(get_array_im((gt_imgs[ind]).convert("L")))
        if len(label.flatten()) != len(pred.flatten()):
            print(
                'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label.flatten()), len(pred.flatten()), gt_imgs[ind],
                    pred_imgs[ind]))
            continue

        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)

        if ind > 0 and ind % 10 == 0:
            print('{:d} / {:d}: mIou-{:0.3f}; mPA-{:0.2f}'.format(ind, len(gt_imgs),
                                                                    100 * np.nanmean(per_class_iu(hist)),
                                                                    100 * np.nanmean(per_class_PA(hist))))
    mIoUs = per_class_iu(hist)
    # print("The mIOUS are {}".format(mIoUs))
    mPA = per_class_PA(hist)
    print(">>>>>>>iIOU Done!")
    # avg = sum(mIoUs) / 10
    # print("The average iIOU result is {}".format(avg))

    for ind_class in range(num_classes):
    # for ind_class in range(1):
        print('===>' + name_classes[ind_class] + ':\tmIoU:' + str(round(mIoUs[ind_class] * 100, 2)) + '; mPA:' + str(
            round(mPA[ind_class] * 100, 2)))

    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)) + '; mPA: ' + str(round(np.nanmean(mPA) * 100, 2)))
    return mIoUs
# Please Note the panoptic prediction is 2 times the size of the original size

compute_mIoU("./SA-Compression/1-mask-generation/seg-gt/", 
 "/home/wang3_y@WMGDS.WMG.WARWICK.AC.UK/Desktop/EfficientPS/output_265_0/recolor_265/", 2, ["0","1"])


# compute_mIoU("./SA-Compression/2-two-stream-gen/GT_IA/", 
#  "./SA-Compression/2-two-stream-gen/IA/264-18/", 2, ["0","1"])
    


#Compute the iIOU of the two diectionaries
# def eval_iIOU(gt_dir, pred_dir, num_classes, name_classes):
#     gt_list = os.listdir(gt_dir)
#     # print(gt_list)
#     gt_list.sort(key=lambda x: int(x.split('.')[0]))  
#     # print(gt_list) 
#     pred_list = os.listdir(pred_dir)
#     pred_list.sort(key=lambda x: int(x.split('.')[0]))
#     # print(pred_list[0:154])

#     total_num = len(gt_list)

#     sum = 0
#     for i in range(0,total_num):
#         iIOU = compute_mIoU(gt_dir, pred_dir,num_classes, name_classes)
#         print("....The {} th image pair's iIOU is {}...".format(i,iIOU))
#         sum += iIOU
    
#     print(">>>>>>>iIOU Done!")
#     avg = sum / total_num
#     print("The average iIOU result is {}".format(avg))

# eval_iIOU("./SA-Compression/1-mask-generation/seg-gt/", 
# "./SA-Compression/1-mask-generation/seg-pred/", 2, "'1','2'")
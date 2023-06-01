import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import collections
import os
import math

# # The original imgs, the seg pred mask, the saved Important area stream frames path, the saved Non-impoatant area stream frames path
# def add_mask2image_binary(images_path, masks_path, masked_path, N_path):
#     img_name_list = os.listdir(images_path)
#     img_name_list.sort(key=lambda x: int(x.split('.')[0]))
#     masks_name_list = os.listdir(masks_path)
#     masks_name_list.sort(key=lambda x: int(x.split('.')[0]))
#     total_num = len(img_name_list)    
#     for i in range(0,total_num):
#         # Add binary masks to images        
#         img = cv2.imread(images_path +  img_name_list[i])
#         print(img.shape) 
#         # mask = cv2.imread(masks_path, cv2.IMREAD_GRAYSCALE)  # 将彩色mask以二值图像形式读取
#         mask = cv2.imread(masks_path + masks_name_list[i],cv2.IMREAD_GRAYSCALE)  # 将彩色mask以二值图像形式读取
#         # cv2.imwrite('/home/haonan_zhao/code/imagecut/mask/gray_mask2.png',mask)
#         masked = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask)  #将image的相素值和mask像素值相加得到结果
#         # print(masked.shape)        
#         cv2.imwrite(masked_path +   masks_name_list[i], masked)
#         print("Storing the {}th IA frame".format(i))       
#         I = img - masked
#         # print(masked.shape) 
#         cv2.imwrite(N_path +  masks_name_list[i], I) 
#         print("Storing the {}th NA frame".format(i))
#     print("...All stream frames generated!...")    


# The original imgs, the GT mask, the saved Important area stream frames path, the saved Non-impoatant area stream frames path
def add_mask2image_binary(images_path, masks_path, masked_path, N_path):
    img_name_list = os.listdir(images_path)
    img_name_list.sort(key=lambda x: int(x.split('.')[0]))
    masks_name_list = os.listdir(masks_path)
    masks_name_list.sort(key=lambda x: int(x.split('.')[0]))
    total_num = len(img_name_list)    
    for i in range(0,total_num):
        # Add binary masks to images        
        img = cv2.imread(images_path +  img_name_list[i])
        # print(img.shape) 
        # mask = cv2.imread(masks_path, cv2.IMREAD_GRAYSCALE)  # 将彩色mask以二值图像形式读取
        mask = cv2.imread(masks_path + masks_name_list[i],cv2.IMREAD_GRAYSCALE)  # 将彩色mask以二值图像形式读取
        # cv2.imwrite('/home/haonan_zhao/code/imagecut/mask/gray_mask2.png',mask)
        masked = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask)  #将image的相素值和mask像素值相加得到结果
        # print(masked.shape)        
        cv2.imwrite(masked_path +   masks_name_list[i], masked)
        print("Storing the {}th IA frame".format(i))       
        I = img - masked
        # print(masked.shape) 
        cv2.imwrite(N_path +  masks_name_list[i], I) 
        print("Storing the {}th NA frame".format(i))
    print("...All stream frames generated!...")      

images_path = "./SA-Compression/0000/"
masks_path = "./SA-Compression/1-mask-generation/gt/"
masked_path = "./SA-Compression/2-two-stream-gen/GT_IA/"
N_path = "./SA-Compression/2-two-stream-gen/GT-BA/"

# add_mask2image_binary(images_path, masks_path, masked_path, N_path)
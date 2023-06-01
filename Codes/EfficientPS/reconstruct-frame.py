import os
# import io

import skimage.io as io
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import transform,data
import skimage.metrics
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR

path_O = "/content/frames/O"

def rgb(img):
    b,g,r = cv2.split(img) 
    img_rgb = cv2.merge([r,g,b])
    return img_rgb

def recon(path_I, path_N,path_recon):
    # The folders of the original frames
    img_I_folder=path_I
    filelist_I = os.listdir(img_I_folder)
    filelist_I.sort(key=lambda x: int(x.split('.')[0]))
    # print(filelist_I)
    img_N_folder=path_N
    filelist_N = os.listdir(img_N_folder)
    filelist_N.sort(key=lambda x: int(x.split('.')[0]))
    # print(filelist_N)

    for i in range(0,len(filelist_I)):
        img_I = cv2.imread(path_I + str(i).zfill(3)+'.png', 1)  # 读取图像
        img_I = rgb(img_I)
        img_N = cv2.imread(path_N + str(i).zfill(3)+'.png', 1)  # 读取图像
        img_N = rgb(img_N)
        img_O = img_I + img_N
        io.imsave(path_recon + '/' + str(i) + '.png', img_O)
        print("Saving" + str(i) + "images now...")

    print("Original frames generated!!!")

# recon("/home/wang3_y@WMGDS.WMG.WARWICK.AC.UK/Desktop/SA-Compression/videos/265-28-I/", 
# "/home/wang3_y@WMGDS.WMG.WARWICK.AC.UK/Desktop/SA-Compression/videos/265-32-N/",
# "/home/wang3_y@WMGDS.WMG.WARWICK.AC.UK/Desktop/SA-Compression/videos/265/28_32/")

def aver_psnr_ssim(path_ori, path_recon):
    # Initialize variables to store the sum of PSNR and SSIM values
    sum_psnr = 0
    sum_ssim = 0
 
    img_O_folder=os.listdir(path_ori)
    img_O_folder.sort(key=lambda x: int(x.split('.')[0]))
    # print( img_O_folder)
    img_R_folder=os.listdir(path_recon)
    img_R_folder.sort(key=lambda x: int(x.split('.')[0]))
    # print( img_R_folder)
    num = len(img_O_folder)
    print(num)

    for i in range(0, num):
        img1 = cv2.imread(path_ori+img_O_folder[i],1)
        img2 = cv2.imread(path_recon+img_R_folder[i],1)
        psnr = PSNR(img1, img2)
        ssim = SSIM(img1, img2, multichannel=True)
        # Add the values to the sum
        sum_psnr += psnr
        sum_ssim += ssim
        
    print(">>>>>>The average PSNR is {}, SSIM is {}".format(sum_psnr/num, sum_ssim/num))

# "./SA-Compression/0000/" "/home/wang3_y@WMGDS.WMG.WARWICK.AC.UK/Desktop/SA-Compression/2-two-stream-gen/IA/", "/home/wang3_y@WMGDS.WMG.WARWICK.AC.UK/Desktop/SA-Compression/2-two-stream-gen/BA/"

aver_psnr_ssim("./SA-Compression/0000/", 
"/home/wang3_y@WMGDS.WMG.WARWICK.AC.UK/Desktop/SA-Compression/videos/265/28_32/")
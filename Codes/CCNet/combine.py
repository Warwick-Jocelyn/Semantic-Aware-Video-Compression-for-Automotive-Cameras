import os
# import io

import skimage.io as io
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import transform,data
 
# The folders of the original frames
img_IA_folder='./Compare'

img_BA_folder='./Compare'

# To get the original name
    
start_frame = 1
end_frame = 296
for i in range(start_frame,end_frame,1):
    BA_name = "BA-265-" + str(i)+ ".png" 
    path_BA = img_BA_folder + "/"+ BA_name
    img_BA = cv2.imread(path_BA)
    
    IA_name = "IA-265-" + str(i)+ ".png" 
    path_IA = img_IA_folder + "/"+ IA_name
    img_IA = cv2.imread(path_IA)
    # To combine the frames
    img_combined1 = img_BA + img_IA
    # To save the frames
    B,G,R = cv2.split(img_combined1)
    img_combined = cv2.merge([R,G,B])
    io.imsave('./combine_23_32_X265/img-265-' + str(i) + '.png', img_combined)
    print("save the image"+ str(i) +".")
 
print("All images saved-----rename !!!")

import os
# import io

import skimage.io as io
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import transform,data
import subprocess

import cv2
import numpy as np
import matplotlib.pyplot as plt
import collections

path_O = "./SA-Compression/0000/"
img_I_folder='./SA-Compression/2-two-stream-gen/IA/'
img_N_folder='./SA-Compression/2-two-stream-gen/BA/'

# 把frames连接成videos！分别进行不同CRF的压缩 X264

#Generating the whole videos using X264
# subprocess.run(["ffmpeg", "-f", "image2", "-i", "path_O+%6d.png", "-vcodec", "libx264", "-r", "10", "path_O+O.mp4"]
# # Compression original video O with crf=23
# subprocess.run([ffmpeg -f image2 -i path_O+O.mp4 -vcodec libx264 -crf 23 path_O+O-264-23.mp4])

# #Generating the IA videos using X264
subprocess.run([ffmpeg -f image2 -i img_I_folder+ %d.png -vcodec libx264 -r 10 img_I_folder+I.mp4])
# # Compression IA videos I with crf=18
subprocess.run([ffmpeg -f image2 -i img_I_folder+I.mp4 -vcodec libx264 -crf 18 img_I_folder+I-264-23.mp4])

ffmpeg -f image2 -i ./SA-Compression/2-two-stream-gen/IA/%d.png -vcodec libx264 -crf 18 -r 10 ./SA-Compression/2-two-stream-gen/IA/I-264-18.mp4

# #Generating the BA videos using X264
# subprocess.run([ffmpeg -f image2 -i img_N_folder+ %d.png -vcodec libx264 -r 10 img_N_folder+N.mp4])
# #Generating the BA videos using crf=27
# subprocess.run([ffmpeg -f image2 -i img_N_folder+N.mp4 -vcodec libx264 -crf 27 img_N_folder+N.mp4-264-27.mp4])

ffmpeg -f image2 -i ./SA-Compression/2-two-stream-gen/BA/%d.png -vcodec libx264 -crf 28 -r 10 ./SA-Compression/2-two-stream-gen/BA/N-264-28.mp4

ffmpeg -f image2 -i ./SA-Compression/2-two-stream-gen/BA/%d.png -vcodec libx265 -crf 32 -r 10 ./SA-Compression/2-two-stream-gen/BA/N-265-32.mp4
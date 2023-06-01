# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 22:55:20 2021

@author: zhaohaonan
"""

#import cv2
#import glob
# 
#fps = 10    #保存视频的FPS，可以适当调整
# 
#fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#videoWriter = cv2.VideoWriter('E:/projectdatasets/mp4_to_png/ir_face.mp4',fourcc,fps,(224,-1))#最后一个是保存图片的尺寸
#imgs=glob.glob('E:/projectdatasets/mp4_to_png/*.png')
#for imgname in imgs:
#    frame = cv2.imread(imgname)
#    videoWriter.write(frame)
#videoWriter.release()

#ffmpeg -i ir_face.mp4 -vf scale=800:-1 ir_face_224171.mp4

import cv2
import argparse
import os


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Process pic')
    parser.add_argument('--input', help='video to process', dest='input', default=None, type=str)
    parser.add_argument('--output', help='pic to store', dest='output', default=None, type=str)
    # default为间隔多少帧截取一张图片；我这里用10刚刚好！
    parser.add_argument('--skip_frame', dest='skip_frame', help='skip number of video', default=1, type=int)
    # input为输入视频的路径 ，output为输出存放图片的路径
    # args = parser.parse_args(['--input', r'./SA-Compression/0000/O-264-18.mp4', r'--output', 'Desktop/SA-Compression/0000/264-18/'])
    args = parser.parse_args(['--input', r'/home/wang3_y@WMGDS.WMG.WARWICK.AC.UK/Desktop/SA-Compression/videos/I-265-28.mp4', r'--output', 
    '/home/wang3_y@WMGDS.WMG.WARWICK.AC.UK/Desktop/SA-Compression/videos/265-28-I/'])
    return args

def process_video(i_video, o_video, num):
    cap = cv2.VideoCapture(i_video)
    num_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    expand_name = '.png'
    sur_name = ''
    if not cap.isOpened():
        print("Please check the path.")
    cnt = 0
    count = 0
    while 1:
        ret, frame = cap.read()
        cnt += 1
 
        if cnt % num == 0:
            count += 1
#            cv2.imwrite(os.path.join(o_video, sur_name + str(count) + expand_name), frame)
            if count < 10:
                cv2.imwrite(os.path.join(o_video, sur_name + str("00") + str(count-1) + expand_name), frame)
            elif count >= 10 and count < 100:
                cv2.imwrite(os.path.join(o_video, sur_name + str("0") + str(count-1) + expand_name), frame)
            else:
                cv2.imwrite(os.path.join(o_video, sur_name + str(count-1) + expand_name), frame)

        if not ret:
            break


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    print('Called with args:')
    print(args)
    process_video(args.input, args.output, args.skip_frame)

# ! ffmpeg -i ./SA-Compression/0000/O-264-18.mp4', r'--output -vcodec libx264 -r 1 ./SA-Compression/264-18/%d.png
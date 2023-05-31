ffmpeg -f image2 -i img%d.png  -vcodec libx264 -r 17 important.mp4

ffmpeg -i Desktop/Warwick_Project/Results/output_00.mp4 -c:v libx264 -crf 23 output_00_23.mp4

sudo apt install x264 x265

AJ3F542c,3jn

ffmpeg -i Original_video_r17.mp4 -vcodec libx264 -r 17 img-%d.jpeg

ffmpeg -i  output_00_23.mp4 -i output_00.mp4 -lavfi libvmaf = “model_path = ‘Users/oneeating/Downloads⁩/⁨anaconda3/pkgs/ffmpeg-4.3.2-h4dad6da_0/⁨bin/vmaf_v0.6.1.json’”  -f null -

# https://github.com/slhck/ffmpeg-quality-metrics
ffmpeg_quality_metrics output_00_07.mp4 output_00.mp4
ffmpeg_quality_metrics output_00_07.mp4 output_00.mp4 -m vmaf

#https://netflixtechblog.com/toward-a-better-quality-metric-for-the-video-community-7ed94e752a30

# the “Video Multi- Method Assessment Fusion” algorithm VMAF [26], developed by Netflix to evaluate video codecs

"""
To generate videos through ffmpeg

"""

# To activate the environment
conda create --name ffmpeg-example

################################################################
conda activate ffmpeg-example

conda install -c conda-forge x264 ffmpeg


# To generate video with no other conditions.
ffmpeg -f image2 -i img%d.png -vcodec libx264 -r 7 Original_video_nopara.mp4

# To generate the frames from the video
ffmpeg -f image2 -i img%d.png -vcodec libx264 -r 17 Original_video_r17.mp4
ffmpeg -i ./Desktop/Results/output_nopara.mp4 img-%4d.png
-vcodec libx264

# To generate video with CRF=17.
ffmpeg -f image2 -i ./Data/img%4d.png ./Data/Original_video_r17.mp4

# To generate video with different fps conditions.
# -r: It means the frame per second
ffmpeg -f image2 -i img%d.png -vcodec libx264 -crf 27 -pix_fmt yuv444p Recon_2327_264_crf27.mp4

ffmpeg -f image2 -i ./data/img%4d.png ./data/Original_video_r4.mp4
ffmpeg -f image2 -i ./data/img%4d.png ./data/Original_video_r8.mp4
ffmpeg -f image2 -i ./data/img%4d.png ./data/Original_video_r16.mp4
ffmpeg -f image2 -i ./data/img%4d.png ./data/Original_video_r32.mp4

# To go thorugh the same H.264 codec
ffmpeg -i ./Data/Original_video_r*.mp4 -c:v libx264 -crf 23 output_00_23.mp4

# To go thorugh the different CRF using the same video

ffmpeg -f image2 -i img%d.png -vcodec libx265 -crf 32 -pix_fmt yuv444p Recon_2832_265_crf32.mp4

ffmpeg -i OF_265_crf28.mp4 img-265-28-%d.png

ffmpeg_quality_metrics Recon_1823_264_crf23.mp4 Original_crf0.mp4


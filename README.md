# Semantic-aware-compression
Codes for the “Semantic-Aware Video Compression for Automotive Cameras”

we propose a semantic-aware (SA) video compression (SAC) framework that compresses separately and simultaneously region-of-interest and region-out-of-interest of automotive camera video frames, before transmitting them to processing unit(s), where the data are used for perception tasks, such as object detection, semantic segmentation, etc. Using our newly proposed technique, the region-of-interest (ROI), encapsulating most of the road stakeholders, retains higher quality using lower compression ratio.

<img src="/doc/Fig-1.png" alt="Illustrating of Semantic-aware Compression (SAC) on the vehicle" width="700"/>


This is the **PyTorch re-implementation** of our TIV paper: 
[Semantic-Aware Video Compression for Automotive Cameras](https://ieeexplore.ieee.org/abstract/document/10103198). 

## Visual Results
Here is a [vedio clip](https://mega.nz/folder/hlJkRARQ#lZoi_3-o7bn3TEOVBO33YA) to show how our ROI streams look like after the segmentation and macroblock processing.

## Requirements
You can use any segmentation model for the stream seperation, here, we show two examples of using the CCNet for the Cityscapes, and the EfficeintPS for the KITTI-Step dataset. 
![Illustrating of separation of ROI and non-ROI streams. ](/doc/Fig-4.png)
- The requirement for CC-Net is from [here](https://github.com/speedinghzl/CCNet#requirements).
- Requirement for EfficientPS is from [here](https://github.com/DeepSceneSeg/EfficientPS#system-requirements).
- Download FFMPEG for X264 or X265 compression [here](https://ffmpeg.org/download.html)

## Dataset and Model

| Method      | Dataset    | Download |
|-------------|------------|----------|
| [CCNet](https://github.com/speedinghzl/CCNet)  | [Cityscapes](https://www.cityscapes-dataset.com/downloads/) | [Model](https://mega.nz/file/9honnLgQ#41ajWjzjc1vbuiEJYVlSfrsmna-fqLEi18q4Sa3qqNE)|
| [EfficientPS](https://github.com/DeepSceneSeg/EfficientPS) | [KITTI-STEP](https://www.cvlibs.net/datasets/kitti/eval_step.php) |[Model](https://www.dropbox.com/s/4z3qiaew8qq7y8n/efficientPS_kitti.zip?dl=0)|

## Run
### Step 1 ROI and non-ROI Stream Generation
run codes with the following cmd:
```
cd ./Codes/CCNet
python Rename_Rescale.py
python test.py
python test.py
python TwoStream_generate.py
```

Note: 
- Change the path from each .py file if needed.
- The CC-Net is re-trained and give results on the four categories like the Fig. 2.
- The ROI and non ROI mask generation from KITTI-STEP require some post-processing to access the labels.  

### Step 2 Compression
Some commonly used cmd for [FFMPEG](https://ffmpeg.org/) can be found from the ./code/compression/., you can change the constant rate factor (-crf) for changing the compression quality. 

There are also lots of parameters such as framerate, codecs that you may adjust acording to your need, having fun play with them !

After generating the two frames of the stream, we can combine the stream after compression with different CRF values.

```
python combine.py
```

### Step 3 Evaluation
```
cd ./Codes/Eva/
python *** 
```


## Citing SAC
If you find this code helpful in your research or wish to refer to the baseline results, please use the following BibTeX entry.

```BibTeX
@article{wang2023semantic,
  title={Semantic-Aware Video Compression for Automotive Cameras},
  author={Wang, Yiting and Chan, Pak Hung and Donzella, Valentina},
  journal={IEEE Transactions on Intelligent Vehicles},
  year={2023},
  publisher={IEEE}
}

```

If you use the CCNet for segmentation, please consider citing
```BibTeX
@inproceedings{huang2019ccnet,
  title={Ccnet: Criss-cross attention for semantic segmentation},
  author={Huang, Zilong and Wang, Xinggang and Huang, Lichao and Huang, Chang and Wei, Yunchao and Liu, Wenyu},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={603--612},
  year={2019}
}

```
If you use the EfficientPS for segmentation, please consider citing
```BibTeX
@article{mohan2021efficientps,
  title={Efficientps: Efficient panoptic segmentation},
  author={Mohan, Rohit and Valada, Abhinav},
  journal={International Journal of Computer Vision},
  volume={129},
  number={5},
  pages={1551--1579},
  year={2021},
  publisher={Springer}
}

```

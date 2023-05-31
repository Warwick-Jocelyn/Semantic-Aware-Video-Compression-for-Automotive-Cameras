# Semantic-aware-compression
Codes for the “Semantic-Aware Video Compression for Automotive Cameras”

we propose a semantic-aware (SA) video compression (SAC) framework that compresses separately and simultaneously region-of-interest and region-out-of-interest of automotive camera video frames, before transmitting them to processing unit(s), where the data are used for perception tasks, such as object detection, semantic segmentation, etc. Using our newly proposed technique, the region-of-interest (ROI), encapsulating most of the road stakeholders, retains higher quality using lower compression ratio.

<img src="/doc/Fig-1.png" alt="Illustrating of Semantic-aware Compression (SAC) on the vehicle" width="700"/>


This is the **PyTorch re-implementation** of our TIV paper: 
[Semantic-Aware Video Compression for Automotive Cameras](https://ieeexplore.ieee.org/abstract/document/10103198). 

## Dataset
The cityscape video can be downloaded from the [official website](https://www.cityscapes-dataset.com/downloads/)
The KITTI-Step video can also be downloaded from the [official website](https://www.cvlibs.net/datasets/kitti/eval_step.php)

## Implementation
We can use any segmentation model for stream seperation, here, we use the CCNet for the Cityscapes, and the EfficeintPS for the KITTI-Step dataset. 
The CC-Net is re-trained with the pre-trained model here:
Here is a [vedio clip](https://mega.nz/folder/hlJkRARQ#lZoi_3-o7bn3TEOVBO33YA) to show how our ROI streams look like after the segmentation and macroblock processing.

![Illustrating of separation of ROI and non-ROI streams. ](/doc/Fig-4.png)





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

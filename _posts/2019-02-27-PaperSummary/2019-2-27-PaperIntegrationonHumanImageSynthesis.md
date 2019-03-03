---
layout: post
title: Paper about Person Image Synthesis
mathjax: true
categories: Paper
tags: Person, Image-Synthesis
keywords: Image-Synthesis
description: 整合一些与人物图像生成有关的论文
mermaid: true
status: Writing
---

# Pose Guided Person Image Generation (PG2)

> 发表：  NIPS 2017  
> Paper: [Pose Guided Person Image Generation](https://arxiv.org/abs/1705.09368)  
> Code: [Pose-Guided-Image-Generation](https://github.com/harshitbansal05/Pose-Guided-Image-Generation)  
> 
> 论文设计了一个两个阶段的网络（同步训练）， 如下图所示，第一个阶段是基于UNet设计的生成器网络，用于人物姿态转换。 第二个阶段为是同样基于UNet设计的GAN模型，用于纹理渲染。 

<img src="https://raw.githubusercontent.com/huangtao36/huangtao36.github.io/master/_posts/2019-02-27-PaperSummary/HumanImageSynthesis/pg2_networks.png" style="zoom:70%" /> 

# Soft-Gated Warping-GAN for Pose-Guided Person Image Synthesis

> 发表： NIPS 2018  
> Paper: [Soft-Gated Warping-GAN for Pose-Guided Person Image Synthesis](https://papers.nips.cc/paper/7329-soft-gated-warping-gan-for-pose-guided-person-image-synthesis.pdf)  
> Code: None  
>   
> 两个阶段的模型，分离训练。 第一阶段用于生成目标图像的 Parser Mask(human segmentation map)，网络结构参考Pix2Pix。  
> 第二阶段生成目标人物图像， 网络结构基于Pix2pixHD。    
> 网络模型的核心在于基于 GEO 设计了一个几何变换的结构，计算原图与目标图像对应的 Parser 数据之间的几何参数，利用这个变换参数对原人物图像的特征图进行了一次 Warping。  

<img src="https://raw.githubusercontent.com/huangtao36/huangtao36.github.io/master/_posts/2019-02-27-PaperSummary/HumanImageSynthesis/softgatewarpinggan_networks.png" style="zoom:70%" /> 

> 附： GEO: Convolutional neural network architecture for geometric matching  
> 发表： CVPR 2017  
> Code: [cnngeometric_pytorch](https://github.com/ignacio-rocco/cnngeometric_pytorch)  
> Website: [https://www.di.ens.fr/willow/research/cnngeometric/](https://www.di.ens.fr/willow/research/cnngeometric/)  
> Paper: [Convolutional neural network architecture for geometric matching](https://arxiv.org/abs/1703.05593)

# Disentangled Person Image Generation

> 发表： CVPR 2018  
> Code: None, [github](https://github.com/charliememory/Disentangled-Person-Image-Generation)  
> Website: [https://homes.esat.kuleuven.be/~liqianma/CVPR18_DPIG/](https://homes.esat.kuleuven.be/~liqianma/CVPR18_DPIG/)  
> Paper: [Disentangled Person Image Generation](http://homes.esat.kuleuven.be/~liqianma/pdf/CVPR18_Ma_Disentangled_Person_Image_Generation.pdf)  
>  


# Everybody Dance Now


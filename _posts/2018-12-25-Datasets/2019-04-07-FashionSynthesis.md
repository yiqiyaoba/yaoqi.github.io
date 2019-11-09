---
layout: post
title: Fashion Synthesis Benchmark  
mathjax: true
categories: Dataset
tags: [Dataset]
keywords: 
description: 
mermaid: true
status: Completed
---

**Fashion Synthesis Benchmark** facilitates the studies of generating new clothing images. It includes 78,979 images selected from the [In-shop Clothes Benchmark](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html). Each image is associated with several sentences as captions and a segmentation map. If you use the **images**, **captions**, and **segmentations**, please appropriately cite the papers of [DeepFashion](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Liu_DeepFashion_Powering_Robust_CVPR_2016_paper.pdf) and [FashionGAN.](https://arxiv.org/pdf/1710.07346.pdf)

[website: http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/FashionSynthesis.html](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/FashionSynthesis.html)

所有数据文件包括： language_original.mat、 ind.mat、G2.h5

其中: 

- ind.mat： 训练（70000）与测试（8979）的数据 id，训练的数据没有配对的， 测试的在8979数量中随机匹配（这里有可能会匹配到男--女）。
- language_original.mat:  
  - cate_new: 服装类别
  - codeJ: 词汇编码
  - color_： 服装颜色
  - engJ： text
  - gender_: 性别
  - nameList： 对应图片文件
  - sleeve_： 长、短、没有袖子




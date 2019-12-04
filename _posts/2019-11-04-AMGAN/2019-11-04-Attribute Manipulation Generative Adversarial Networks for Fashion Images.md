---
layout: post
title: Attribute Manipulation Generative Adversarial Networks for Fashion Images(ICCV2019)
mathjax: true
categories: Paper
tags: [Paper]
keywords: Fashion
description: 
---

> 基于 Attribute 来编辑人物时尚图像中上衣的颜色以及袖长.

<img src="https://raw.githubusercontent.com/huangtao36/huangtao36.github.io/master/_posts/which_dir/xxx.png" style="zoom:80%" />

## Abstract

基于摘要的描述，获取到一下信息：

- 单一生成器完成多领域（multi-domain）图像翻译。

- 最相关的论文是 [GANimation(ECCV2018)]( http://openaccess.thecvf.com/content_ECCV_2018/papers/Albert_Pumarola_Anatomically_Coherent_Facial_ECCV_2018_paper.pdf ) 和 [SaGAN(ECCV2018)]( http://openaccess.thecvf.com/content_ECCV_2018/papers/Gang_Zhang_Generative_Adversarial_Network_ECCV_2018_paper.pdf ) ，这两个方法基于注意力机制编辑属性相关区域。

  > 这两个方法都是做面部特征转换的。GANimation 通过类似于 StarGAN 的网络结构使用两个生成器分别生成一个 RGB 图像和一个 Map (which describes in a continuous manifold the anatomical facial movements defining a human expression. )， 然后整合 RGB 图像和这个 Map 得到最终的结果。 SaGAN  引入空间注意力机制来避免模型更改与要求属性无关的区域，生成器由两个部分组成：编辑人脸图像的属性操作网络（AMN），定位属性特定区域的空间关注网络（SAN）。

- 之前的方法当属性数量增加时效果不佳，因为 Attention Mask 的训练取决于分类损失。
- 设计 AMGAN 解决这个问题，AMGAN的生成器网络使用类激活图(class activation maps，CAM)来增强其注意力机制，也通过基于属性相似性分配参考图像来利用感知损失。(具体什么意思？继续看原文)
- 多加一个判别器，专门检测与属性相关的区域
- AMGAN 可以操作指定区域的属性
- 评估指标有传统的，也有基于图像检索的方法

## Introduction

	> 按每段的内容进行大概的介绍

​	Attribute manipulation 的任务概述，并引入到时尚图像方面。同时介绍应用方向。除了可编辑不满意的图像的属性外，一些研究也在将编辑属性后的图像用于检索任务。

​		介绍了 [StarGAN(CVPR2018)](http://openaccess.thecvf.com/content_cvpr_2018/papers/Choi_StarGAN_Unified_Generative_CVPR_2018_paper.pdf),  [GANimation(ECCV2018)](http://openaccess.thecvf.com/content_ECCV_2018/papers/Albert_Pumarola_Anatomically_Coherent_Facial_ECCV_2018_paper.pdf) 和 [SaGAN(ECCV2018)](http://openaccess.thecvf.com/content_ECCV_2018/papers/Gang_Zhang_Generative_Adversarial_Network_ECCV_2018_paper.pdf) , 也是本文对比的三个算法。

​		具体介绍本文的任务， 在 [Deepfashion](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Liu_DeepFashion_Powering_Robust_CVPR_2016_paper.pdf) 和 [Shopping100](https://ieeexplore.ieee.org/abstract/document/8354290)上做实验, Shopping100 似乎是非公开的数据集。

<img src="https://raw.githubusercontent.com/huangtao36/huangtao36.github.io/master/_posts/2019-11-04-AMGAN/assets/1575425133836.png" alt="1575425133836" style="zoom:80%;" />








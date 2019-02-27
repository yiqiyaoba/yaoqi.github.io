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
> 论文设计了一个两个阶段的网络（同步训练）， 如下图所示，第一个阶段基于UNet设计的生成器网络，用于人物姿态转换。 第二个阶段为GAN模型，生成器同样基于UNet结构，用于纹理渲染。 

<img src="https://raw.githubusercontent.com/huangtao36/huangtao36.github.io/master/_posts/2019-02-27-PaperSummary/HumanImageSynthesis/pg2_networks.png" style="zoom:50%" /> 


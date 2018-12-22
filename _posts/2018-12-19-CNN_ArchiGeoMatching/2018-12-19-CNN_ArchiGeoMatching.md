---
layout: post
title: Convolutional neural network architecture for geometric matching(CVPR'17)
mathjax: true
categories: Paper
tags: GeometricMatch, SpatialTransfer
keywords: geometric_matching, spatial_transfer
description: Paper Reading Notes
mermaid: true
status: Writing
---

> **资料**：  
> Paper: [Convolutional neural network architecture for geometric matching](https://arxiv.org/abs/1703.05593) (CVPR'17)   
> Website: [https://www.di.ens.fr/willow/research/cnngeometric/](https://www.di.ens.fr/willow/research/cnngeometric/)  
> Code: [Pytorch Code](https://github.com/ignacio-rocco/cnngeometric_pytorch)

---

# Paper Work
这是关于图像几何/特征匹配的工作，如下图所示：

<img src="https://raw.githubusercontent.com/huangtao36/huangtao36.github.io/master/_posts/2018-12-19-CNN_ArchiGeoMatching/assets/ImageGeometricMatching.png" style="zoom:70%" />

给定一张图片 A，图中包含一个物体（这里是摩托车）， 以另一张包含相同特征物体的图像 B 作为目标， 实现的是 A 中的摩托车通过仿射变换拟合 B 中的图像。 学习的是这个仿射变换过程中的仿射参数。

扩大范围来说，这是估计图像之间对应关系的工作，是计算机视觉中的一个基本问题。可应用于三维重建、图像增强、语义分割等，也有 Paper 将其应用于姿态转换的问题上([Soft-Gated Warping-GAN for Pose-Guided Person Image Synthesis](https://arxiv.org/abs/1810.11610))。

# Contribution

**handle large changes of appearance between the matched images.**
  
经典的相似度估计方法，比如使用 SIFT 或 HOG 获取局部特征丢弃不正确的匹配进行模糊匹配，然后将模糊匹配的结果输入到 RANSAC 或者 Hough transform 中进行精确匹配，虽然效果不错但是无法应对场景变换较大以及复杂的几何形变的情况。 本文使用CNN提取特征以应对这两点不足:

- 用CNN特征替换原有经典特征，即使场景变换很大，也能够很好的提取特征；
- 设计一个匹配和变换估计层，加强模型鲁棒性。

(from [jianshu huyuanda](https://www.jianshu.com/p/837615ee36fd))

# Architecture/Method

网络结构如下图所示：
<img src="https://raw.githubusercontent.com/huangtao36/huangtao36.github.io/master/_posts/2018-12-19-CNN_ArchiGeoMatching/assets/architecture.png" style="zoom:70%" />

**输入：** $I_A$, $I_B$, 图片  
**输出：** 仿射参数（参数个数可调整）

**Feature extraction CNN：**  
使用的是VGG-16 crop at pool4, 输出的每一个 feature 都经过了 L2-normalization, 之后得到 $f_A$, $f_B$.
 
**Matching:**  
Matching的方法如下图所示：

<img src="https://raw.githubusercontent.com/huangtao36/huangtao36.github.io/master/_posts/2018-12-19-CNN_ArchiGeoMatching/assets/correlationLayer.png" style="zoom:50%" />

$$
{c_{AB}}(i,j,k) = {f_B}{(i,j)^T}{f_A}({i_k},{j_k})
$$

$f_A$ 与 $f_B$ 进行点乘得到 correlation map($c_AB$)
原来两个 $w×h$的 feature map ，每个 $1×1×d$ 的向量通过点乘得到 $w×h×(w×h)$ 这样一个立方体。立方体当中的每一个位置 $(i，j)$ 表示 $f_B$ 中的$(i，j)$ 位置的点对应 $f_A$ 中所有点的相似度。这里 correlation map 的深度 $(w×h)$ 即 $f_A$ 中所有点被展开成 $k$，表示 $f_A$ 中点的索引。

# Experiments

# Results

# Related Work
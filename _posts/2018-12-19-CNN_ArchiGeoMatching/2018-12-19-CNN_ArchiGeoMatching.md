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

# 资料 
 
> Paper: [Convolutional neural network architecture for geometric matching](https://arxiv.org/abs/1703.05593) (CVPR'17)   
> Website: [https://www.di.ens.fr/willow/research/cnngeometric/](https://www.di.ens.fr/willow/research/cnngeometric/)  
> Code: [Pytorch Code](https://github.com/ignacio-rocco/cnngeometric_pytorch)

# Paper Note

## Paper Work
这是关于图像几何/特征匹配的工作，如下图所示：

<img src="https://raw.githubusercontent.com/huangtao36/huangtao36.github.io/master/_posts/2018-12-19-CNN_ArchiGeoMatching/assets/ImageGeometricMatching.png" style="zoom:70%" />

给定一张图片 A，图中包含一个物体（这里是摩托车）， 以另一张包含相同特征物体的图像 B 作为目标， 实现的是 A 中的摩托车通过变换拟合 B 中的图像。 学习的是这个变换过程中的参数。

扩大范围来说，这是估计图像之间对应关系的工作，是计算机视觉中的一个基本问题。可应用于三维重建、图像增强、语义分割等，也有 Paper 将其应用于姿态转换的问题上([Soft-Gated Warping-GAN for Pose-Guided Person Image Synthesis](https://arxiv.org/abs/1810.11610))。

## Contribution

**handle large changes of appearance between the matched images.**
  
经典的相似度估计方法，比如使用 SIFT 或 HOG 获取局部特征丢弃不正确的匹配进行模糊匹配，然后将模糊匹配的结果输入到 RANSAC 或者 Hough transform 中进行精确匹配，虽然效果不错但是无法应对场景变换较大以及复杂的几何形变的情况。 本文使用CNN提取特征以应对这两点不足:

- 用CNN特征替换原有经典特征，即使场景变换很大，也能够很好的提取特征；
- 设计一个匹配和变换估计层，加强模型鲁棒性。

(from [jianshu huyuanda](https://www.jianshu.com/p/837615ee36fd))

## Architecture/Method

网络结构如下图所示：
<img src="https://raw.githubusercontent.com/huangtao36/huangtao36.github.io/master/_posts/2018-12-19-CNN_ArchiGeoMatching/assets/architecture.png" style="zoom:60%" />

**输入：** $I_A$, $I_B$, 图片  
**输出：** 变换参数（参数个数可调整）

### Feature extraction CNN 
使用的是 VGG-16 crop at pool4, 输出的每一个 feature 都经过了 L2-normalization, 之后得到 $f_A$, $f_B$.
 
### Matching
Matching的方法如下图所示：

<img src="https://raw.githubusercontent.com/huangtao36/huangtao36.github.io/master/_posts/2018-12-19-CNN_ArchiGeoMatching/assets/correlationLayer.png" style="zoom:60%" />

公式如下： 

$$
{C_{AB}}(i,j,k) = {f_B}{(i,j)^T}{f_A}({i_k},{j_k})
$$

原文： where $(i, j)$ and $(i_k, j_k)$ indicate the individual feature positions in the $h×w$ dense feature maps, and $k = h(j_k−1)+i_k$ is an auxiliary indexing variable for $(i_k, j_k)$.

$f_A$ 与 $f_B$ 通过点乘得到 correlation map(${C_{AB}}$)。 结合公式与图，$f_B$ 中的每一个 $1 \times 1 \times d$ 向量， 都乘以了 $f_A$ 中每一个 $1 \times 1 \times d$ 向量， 得到的结果是 ${C_{AB}}$ 当中的每一个位置 $(i，j)$ 表示 $f_B$ 中的 $(i，j)$ 位置的点对应 $f_A$ 中所有点的相似度(这里的‘点’均表示 $1 \times 1 \times d$ 向量)。 

**实现代码如下：**  

```python
# implement by Pytorch
class FeatureCorrelation(torch.nn.Module):
    def __init__(self):
        super(FeatureCorrelation, self).__init__()
    
    def forward(self, feature_A, feature_B):
        b, c, h, w = feature_A.size()
        # reshape features for matrix multiplication
        feature_A = feature_A.transpose(2, 3).contiguous().view(b, c, h * w)
        feature_B = feature_B.view(b, c, h * w).transpose(1, 2)
        # perform matrix mult.
        feature_mul = torch.bmm(feature_B, feature_A)
        correlation_tensor = feature_mul.view(b, h, w, h*w).transpose(2, 3).transpose(1, 2)
        return correlation_tensor
'''
# 输入： feature_A， feature_B， size is（batch, d, h, w）
# 输出： correlation_tensor， size is (batch, (w * h), h, w)

# 其中： .contiguous(): view只能用在contiguous的variable上。
# 如果在view之前用了transpose, permute等，需要用contiguous()来返回一个contiguous copy。
'''
```

通过 Matching 得到的 ${f_{AB}}$ 经过了 RELU 后，再经过 L2-normalization 才得到最后的结果。

>
> **Q: 为什么要经过 RELU 和 L2-normalization？**
> 
> Ans:  First, let us consider the case when descriptor fB correlates well with only a single feature in fA. In this case, the normalization will amplify the score of the match, akin to the nearest neighbor matching in classical geometry estimation. Second, in the case of the descriptor fB matching multiple features in fA due to the existence of clutter or repetitive patterns, matching scores will be down-weighted similarly to the second nearest neighbor test [38].(原文)
>
> 38: D. G. Lowe. Distinctive image features from scale-invariant keypoints. IJCV, 2004.
>
> 注： 我还不了解具体的意思，做个记录备查！ 原文中有更多的 Discussion.  
> huyuanda 的笔记中有简单的介绍： [https://www.jianshu.com/p/837615ee36fd](https://www.jianshu.com/p/837615ee36fd)

### Regression
Regression Network 是得到最后变换参数的网络，如下图所示： 

<img src="https://raw.githubusercontent.com/huangtao36/huangtao36.github.io/master/_posts/2018-12-19-CNN_ArchiGeoMatching/assets/RegressionNetwork.png" style="zoom:60%" />

使用卷积层而不是直接使用全连接层，是因为 as the input correlation map size is quadratic in the number of image features, such a network would be hard to train due to a large number of parameters that would need to be learned, and it would not be scalable due to occupying too much memory and being too slow to use.

最后得到的 $\hat \theta$ 的参数个数是可调的。

### Full Network
整体的网络结构由上述相同的两个结构组成，只是最后的变换参数的自由度（个数）不同，如下图所示，Stage I 输出的是仿射参数 ${\hat \theta _{Aff}}$， 为 **6** 个自由度， 而 Stage II 输出的是 Thin-plate Spline 变换参数 ${\hat \theta _{TPS}}$， 为 **18** 个自由度。

<img src="https://raw.githubusercontent.com/huangtao36/huangtao36.github.io/master/_posts/2018-12-19-CNN_ArchiGeoMatching/assets/architecture_all.png" style="zoom:60%" />

#### Affine Transformation
仿射变换，参数为 6 个自由度。 

实现方法可参考 Pytorch 的 [Spatial Transformer Networks Tutorial](https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html)


#### Thin-plate Spline Transformation
薄板样条插值（Thin-Plate Spline）

> 参考资料：   
> Blog: [数值方法——薄板样条插值（Thin-Plate Spline）](https://blog.csdn.net/VictoriaW/article/details/70161180)
> Paper: [Principal warps: Thin-plate splines and the decomposition of deformations](http://user.engineering.uiowa.edu/~aip/papers/bookstein-89.pdf)

## Experiments

### Loss Function

Loss 计算的是每个栅格点使用*预测参数*和*真实参数*进行变换后得到的值之间的平方距离。

公式如下：

<img src="https://raw.githubusercontent.com/huangtao36/huangtao36.github.io/master/_posts/2018-12-19-CNN_ArchiGeoMatching/assets/loss.png" style="zoom:30%" />

### Training Dataset
由于没有公开的数据集，故使用如下的方法人工构建：

<img src="https://raw.githubusercontent.com/huangtao36/huangtao36.github.io/master/_posts/2018-12-19-CNN_ArchiGeoMatching/assets/SyntheticImageGeneration.png" style="zoom:60%" />

为了避免变换后带来的图像的边界问题，在原始图像中央截取Padded image。
在padded image的中央截取ImageA。
对padded image进行变换，在中央截取相同大小，获得ImageB。

使用的数据集有 Tokyo Time Machine dataset 和 Pascal VOC 2011， 具体请看原文。

### Evaluation Dataset
方法的定量评估是在 Propsal Flow 数据集（[Paper](https://arxiv.org/pdf/1511.05065.pdf)）上做的，这个数据集是用于两幅图像之间关键点匹配的，包含900个图片 pairs，每一个 pairs 都是同一类的不同实例。

### Performance Measure

采用 Propsal Flow 中的关键点匹配的评估方法，经过变换的图像的关键点与目标的关键点的匹配百分比。 匹配与否使用 $\alpha  \cdot \max (h,w),\alpha  = 0.1$ 来确定，在此范围内则认为是匹配的。


## Results
请看论文。

## Related Work
原作者论文： [End-to-end weakly-supervised semantic alignment](http://openaccess.thecvf.com/content_cvpr_2018/papers/Rocco_End-to-End_Weakly-Supervised_Semantic_CVPR_2018_paper.pdf) —— CVPR2018
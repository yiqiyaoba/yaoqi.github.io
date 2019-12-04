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

## Abstract

基于摘要的描述，获取到以下信息：

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

- Attribute manipulation 的任务概述，并引入到时尚图像方面。同时介绍应用方向。除了可编辑不满意的图像的属性外，一些研究也在将编辑属性后的图像用于检索任务。
- 介绍了 [StarGAN(CVPR2018)](http://openaccess.thecvf.com/content_cvpr_2018/papers/Choi_StarGAN_Unified_Generative_CVPR_2018_paper.pdf),  [GANimation(ECCV2018)](http://openaccess.thecvf.com/content_ECCV_2018/papers/Albert_Pumarola_Anatomically_Coherent_Facial_ECCV_2018_paper.pdf) 和 [SaGAN(ECCV2018)](http://openaccess.thecvf.com/content_ECCV_2018/papers/Gang_Zhang_Generative_Adversarial_Network_ECCV_2018_paper.pdf) , 也是本文对比的三个算法。
- 具体介绍本文的任务， 在 [Deepfashion](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Liu_DeepFashion_Powering_Robust_CVPR_2016_paper.pdf) 和 [Shopping100](https://ieeexplore.ieee.org/abstract/document/8354290)上做实验(Shopping100 似乎是非公开的数据集)

<img src="https://raw.githubusercontent.com/huangtao36/huangtao36.github.io/master/_posts/2019-12-04-AMGAN/assets/1575425133836.png" alt="1575425133836" style="zoom:50%;" />

- AMGAN 的具体结构：
  - 引入了一种用于属性操纵的注意机制，而无需利用关于属性位置的任何信息
  - CNN 中提取类激活图（CAM）用于正确定位属性的区分区域。
  - 将CAM用作注意力损失，AMGAN的生成器网络生成注意掩码，从而提高其属性操纵能力。
  - AMGAN使用附加的判别网络来关注属性相关区域，以检测不切实际的属性操作以提高图像翻译性能。
- 本文的方法不使用配对的图像（相关的任务都没有配对的图像）， 因此也无法直接使用 perceptual losses。 本文基于属性相似性获得一个参考图像来解决这个问题。同时，AMGAN 可以通过加入一个 Attention Mask 来编辑特定的区域属性。
- 主要贡献：
  - 利用从同一CNN中提取的CAM增强生成网络的注意力机制，该CAM用于基于属性相似性实现感知损失。
  - 合并一个附加的判别器，重点放在与属性相关的区域。
  - 在特定区域启用属性操作。
  - 在两个时尚数据集上实验，以展示 AMGAN 优于最新方法的性能。 我们还介绍了一种基于图像检索的新方法来测试属性操纵的成功。

## AMGAN

> 重点解决通过前面描述还无法完全理解的内容

**Model**：

模型的结构是基于 StarGAN 修改的。

<img src="https://raw.githubusercontent.com/huangtao36/huangtao36.github.io/master/_posts/2019-12-04-AMGAN/assets/1575427154341.png" alt="1575427097321" style="zoom:60%;" />

整个模型包含一个生成器 $G$ 和两个判别器 $D_I$ 和 $D_C$ ，$G$ 输出： generated image $z$ 和 attention mask $\alpha$. CNN + CAM 是一个与训练好的注意力图生成模型。

其中Attribute 表示为:
$$
m=\{m_1,\dots,m_N,r\}
$$

> where $N$ is the number of attribute values (e.g., long sleeve, red color, etc.) and $r$ indicates the attribute that is being manipulated (e.g., sleeve, color, etc.)

**关于裁剪图像的位置确定方法：**

图中基于 $\alpha$ 将原图与生成的图像裁剪对应的区域用于作为 $D_C$ 判别器的输入。做法是： pixel values of $\alpha$ that are above $50\%$ of its maximum value are segmented followed by estimating a bounding box that covers the largest connected region. Using bounding boxes, $x_C^∗$, $x_C$ are cropped from $x_I^∗$, $x_I$. 更为具体的，怎么确定 bounding boxes， 文中没有更详细的说明。

**判别器优化：**

- Adversarial Loss.

  <img src="https://raw.githubusercontent.com/huangtao36/huangtao36.github.io/master/_posts/2019-12-04-AMGAN/assets//1575440605432.png" alt="1575440605432" style="zoom: 67%;" />

  > 其中 $\lambda_{gp}$ 为 [gradient penalty]( https://arxiv.org/abs/1704.00028 ) 项。另整理笔记探究 gradient penalty 。

- Classification Loss 及 总的 Loss.

  <img src="https://raw.githubusercontent.com/huangtao36/huangtao36.github.io/master/_posts/2019-12-04-AMGAN/assets/1575440675567.png" alt="1575440675567" style="zoom: 67%;" />

**生成器优化：**

生成器优化共包括5个损失： 1-Adversarial loss, 2-Classification loss, 3-Cycle Consistency loss, 4-Attention loss, 5-Perceptual loss.   

总的：  
$$
L_G=L_{adv}^G+\lambda_{cls}L_{cls}^G+\lambda_{cyc}L_{cyc}^G+\lambda_{a}L_{a}^G+\lambda_{p}L_{p}^G
$$

> 其中： $\lambda_{cls} =1, \lambda_{cyc} = 10, \lambda_{a} = 10, \lambda_{p} = 20.$

- **Adversarial loss:**

  <img src="https://raw.githubusercontent.com/huangtao36/huangtao36.github.io/master/_posts/2019-12-04-AMGAN//assets/1575441774611.png" alt="1575441774611" style="zoom: 80%;" />

- **Classification loss:**

  交叉熵损失，与 Adversarial loss 使用同一个模型，只是最后一层不同。

  <img src="https://raw.githubusercontent.com/huangtao36/huangtao36.github.io/master/_posts/2019-12-04-AMGAN//assets/1575441846650.png" alt="1575441846650" style="zoom:80%;" />

- **Cycle Consistency loss:**

  StarGAN 的机制本身是基于 CycleGAN 的，使用原Attribute + 生成的图像输入模型生成与原图同样属性的图像，计算 Cycle Consistency loss。

  <img src="https://raw.githubusercontent.com/huangtao36/huangtao36.github.io/master/_posts/2019-12-04-AMGAN//assets/1575441965255.png" alt="1575441965255" style="zoom:80%;" />

- **Attention loss:**

  生成器生成人物图像的同时生成了一个注意力图。本文使用一个预训练好的 CNN+CAM 模型生成一个注意力图的 groundtruth， 计算 $l_1$ 损失。

  <img src="https://raw.githubusercontent.com/huangtao36/huangtao36.github.io/master/_posts/2019-12-04-AMGAN//assets/1575441989697.png" alt="1575441989697" style="zoom:80%;" />

  **Class Activation Mapping(CAM):**

  相关 Paper: [Learning Deep Features for Discriminative Localization(CVPR2016)](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf). 一个定位图像中符合标签内容位置的工作。

  <img src="https://raw.githubusercontent.com/huangtao36/huangtao36.github.io/master/_posts/2019-12-04-AMGAN//assets/1575442945259.png" alt="1575442945259" style="zoom:60%;" />

- **Perceptual loss:**

  由于没有配对的图像，本文通过寻找一个与生成图像属性相同的图像来计算 loss.

  <img src="https://raw.githubusercontent.com/huangtao36/huangtao36.github.io/master/_posts/2019-12-04-AMGAN//assets/1575442008972.png" alt="1575442008972" style="zoom:80%;" />

## Region-speciﬁc Attribute Manipulation

本文的方法还适用于只改变某个特定的部分，在这里特指只改变袖子的颜色（之前的都是改变整个上衣的颜色）。

<img src="https://raw.githubusercontent.com/huangtao36/huangtao36.github.io/master/_posts/2019-12-04-AMGAN/assets/1575443917438.png" alt="1575443917438" style="zoom:80%;" />

将颜色和袖长的属性生成器生成的注意力图结合即可达到这个效果。

##  Experiments

**模型输入:** [batch, 3+N+M, 128, 128], 3为图像rgb, N 为attribute value的个数（比如红、黄、蓝、 …）， M为 Attribute 的个数（比如：颜色、袖子、…）

**对比算法：**  [StarGAN(CVPR2018)](http://openaccess.thecvf.com/content_cvpr_2018/papers/Choi_StarGAN_Unified_Generative_CVPR_2018_paper.pdf),  [GANimation(ECCV2018)](http://openaccess.thecvf.com/content_ECCV_2018/papers/Albert_Pumarola_Anatomically_Coherent_Facial_ECCV_2018_paper.pdf) 和 [SaGAN(ECCV2018)](http://openaccess.thecvf.com/content_ECCV_2018/papers/Gang_Zhang_Generative_Adversarial_Network_ECCV_2018_paper.pdf)

**数据集：**

- DeepFashion-Synthesis： 78,979张人物图像，128*128， Attribute: color (17), sleeve (4) 
- Shopping 100K: 101,021张服装图像，同作者在18年建立的数据集，未公开，6 attributes: collar
  (17), color (19), fastening (9), pattern (16), sleeve length (9)

**Evaluation Metrics：**

- Classiﬁcation Accuracy：

  使用 ResNet-50 训练一个属性分类器来评价生成图像是否符合属性要求

- Top-k Retrieval Accuracy.：

  Top-k检索准确性考虑了搜索算法是否在Top-k结果中找到正确的图像。 如果检索到的图像包含输入和属性操作所需的属性，则它将为命中“ 1”，否则为未命中“ 0”。

- User Study.：

   20 participants
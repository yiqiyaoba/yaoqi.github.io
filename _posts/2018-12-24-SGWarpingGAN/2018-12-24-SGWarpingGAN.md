---
layout: post
title: Soft-Gated Warping-GAN for Pose-Guided Person Image Synthesis(NeurIPS'18)
mathjax: true
categories: Paper
tags: PoseTransfer, GAN
keywords:
description: Synthesizing person images conditioned on arbitrary poses
mermaid: true
status: Writing
---

# 资料
> Paper: [Soft-Gated Warping-GAN for Pose-Guided Person Image Synthesis](https://papers.nips.cc/paper/7329-soft-gated-warping-gan-for-pose-guided-person-image-synthesis.pdf)  
> Supplementary: [Download(Zip)](http://papers.nips.cc/paper/7329-soft-gated-warping-gan-for-pose-guided-person-image-synthesis-supplemental.zip)， 一些实验结果展示

# Paper Note
## Data Preprocessing

### Parser Label
使用预训练好的模型来生成 Parser 数据。  
可选择的模型有两个： [LIP_SSL](https://github.com/Engineering-Course/LIP_SSL) 和 [LIP_JPPNet](https://github.com/Engineering-Course/LIP_JPPNet),   
这里使用 **LIP_JPPNet** 的原代码，除了重新设计接口函数，其他不经任何改动。


```python
from parseing_generate import generate_parser
```


```python
img_dir = './datasets/examples'
save_dir = './datasets/output/'
```

输入的图片放置在 examples 文件夹内，文件名任意，大小任意。  
每张图片有两个输出，一个是单层的 label，一个是 rgb-3 层的(为了直观显示)， png格式


```python
# Generate parser image
generate_parser(img_dir, save_dir)
```


```python
rgb_im = Image.open(os.path.join(img_dir, "images", '114317_456748.jpg'))
par_im = np.array(Image.open(os.path.join(save_dir, '114317_456748.png')))
plt.subplot(121)
plt.imshow(rgb_im)
plt.subplot(122)
plt.imshow(par_im)
plt.show()
```


![png](output_8_0.png)


输出的 par_im 是单层的 label, 身体每一个部分有固定的编号，编号既是图像中对应的像素值。
```python
1.Hat	帽子     
2.Hair	头发          
3.Glove	手套	
4.Sunglasses	墨镜    
5.UpperClothes	上衣 
6.Dress	连衣裙    
7.Coat	外套          
8.Socks 袜子
9.Pants 裤子     
10.Torso-skin 躯干皮肤    
11.Scarf 围巾
12.Skirt 短裙    
13.Face  脸
14.Left-arm 	左臂    
15.Right-arm	右臂
16.Left-leg    左腿    
17.Right-leg	右腿
18.Left-shoe	左鞋    
19.Right-shoe	右鞋
```

---
需要注意的是，论文中得到的 Parser label 是 20 层的，每一层表示一个身体部分，以 one-hot 形式编码， 顺序如上， 第 0 层是 background 层。


**资源：**  
- LIP 的笔记： [MyNote](https://huangtao36.github.io/dataset/LIP.html)  

---

### Pose Heatmaps
基本的方法是使用 [OpenPose](https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation) 预训练好的模型来生成 heatmaps。

因为 [PG2](https://github.com/harshitbansal05/Pose-Guided-Image-Generation) 提供了预处理过的数据集，这里就直接使用了。之后会封装 OpenPose 的生成代码，方便使用自己的数据（<span class="burk">Note to myself</span>）

**Heatmaps:**  encode the pose with 18 heatmaps. Each heatmap has one point that is ﬁlled with 1 in 4-pixel radius circle and 0 elsewhere.  
需要注意的是， PG2 的 dataloader 得到的 heatmap 是 -1 elsewhere.


```python
from PG2_data_loader import get_loader
```


```python
pose_loader = get_loader('./datasets/DF_img_pose', 1)
```

    _get_train_all_pn_pairs finish ...
    p_pairs length:97854
    _get_train_all_pn_pairs finish ...
    p_pairs length:77538



```python
for step, example in enumerate(pose_loader):
    image_0, image_1, pose_1, mask_1 = example
    break
```


```python
im_cache = pose_1[0, 0, :, :]
plt.subplot(121)
plt.imshow(im_cache)
plt.subplot(122)
plt.imshow(im_cache[30:45, 140:155])
plt.show()
```


![png](output_14_0.png)


Heapmap 对应人体 keypoints 位置：

```python
{0,   "nose"},    {1,  "neck"},    {2,  "Rshoulder"},    
{3,  "Relbow"},   {4,  "Rwrist"},  {5,  "Lshoulder"},    
{6,  "Lelbow"},   {7,  "Lwrist"},  {8,  "Rhip"},         
{9, "Rknee"},    {10, "Rankle"},   {11, "Lhip"},         
{12, "Lknee"},    {13, "Lankle"},   {14, "Reye"},         
{15, "Leye"},     {16, "Rear"},    {17, "Lear"},
```

## Model

### Stage I: Pose-Guided Parsing
![image.png](attachment:image.png)

这里的 Parser 是1.1节中预训练的模型，Encoder-->Decoder 参考 [Pix2pix](https://github.com/phillipi/pix2pix) 的结构(9 residual blocks)。

**Input：**   
- Condition Parsing: [batch, 20, h, w], one-hot  
- Target Pose: [batch, 18, h, w], one-hot  

**Output:**  
- Synthesized Parsing: [batch, 20, h, w], one-hot

---

<span class="burk">Q: Stage I 和 Stage II 是同步训练的还是分开训练的？？？</span>  
Ans: 根据其 loss 的设计来看，应是同步训练的。  
为满足 Test 的过程中没有对应 Target Pose 的 RGB 图片，Stage I 必不可少（没有rgb图片就没办法用LIP的模型生成 Parser 图片），
但其实两个 Stage 也是可分的，可以完全独立的实验。具体效果怎么样，就得实验后才知道了。

---

### Stage II: Warping-GAN Rendering
![image.png](attachment:image.png)

#### Geometrix Matcher
![image.png](attachment:image.png)

Geometrix Matcher 的思想来源于 [GEO](https://arxiv.org/abs/1703.05593), 一个基于图像特征做几何匹配变换的工作。

- Feature Extarctor 是一个预训练好的 VGG16, crop at pool4,  
- Matching Layer 是把两个输入的 Feature 进行点乘，结合到一起，具体可以看 [My Note about GEO](https://huangtao36.github.io/paper/CNN_ArchiGeoMatching.html)。  
- Regression Newwork 是两层卷积层再加上线性层，最后得到变换参数。（使用 F.affine_grid 可得到 Transformation Grid）  
<span class="burk">affine 和 Thin-plate Spline Transformation</span>

#### Soft-gated Warping-Block
![image.png](attachment:image.png)


## Loss
四个Loss

- $L_{adv}$ : 原始的 GAN Loss
- $L_{pivel}$ : Pixel-wise softmax loss, 用于 Stage I. 来源：[Skeleton-aided Articulated Motion Generation](https://arxiv.org/pdf/1707.01058.pdf), 这个来源是论文中的引用，但似乎没有解释这个loss的具体含义，参考 [Blog](https://blog.csdn.net/magua1993/article/details/78230100) 中的解释，pixel-wise loss强调的是两幅图像之间每个对应像素的匹配，这与人眼的感知结果有所区别。通过pixel-wise loss训练的图片通常会较为平滑，缺少高频信息。还有些不知所云，<span class="burk">待深入了解</span>。
- $L_{perceptual}$ : 常用于图像精细化，超分辨率的一个loss。 来源：[Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/pdf/1603.08155.pdf), 简单的理解就是比较的两个图像经过同一个预训练好的网络（比如VGG-16）,去一个中间的特征层输出，去计算这两个特征矩阵之间的均方误差。 
- $L_{PH}$ : pyramidal hierarchy loss. 如下图：

![image.png](attachment:image.png)

$L_{PH}$ 计算的是 real 和 fake 图像经过判别器时对不同的特征层的输出计算loss的结果。
$${L_{PH}} = {\sum\limits_{i = 0}^n {{a_i}\left\| {{F_i}(\hat I) - {F_i}(I)} \right\|} _1}$$

## Experiments

### Dataset
#### DeepFashion
consists of 52,712 person images in fashion clothes with image size 256×256.
[website](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html), 我一直没有成功下载过，如果想用，可以考虑使用 [PG2](https://github.com/harshitbansal05/Pose-Guided-Image-Generation) 中提供下载的数据集，是否预处理过有待验证。

#### Market-1501
contains 322,668 images collected from 1,501 persons with image size 128×64.  
暂未做具体了解。

### Evaluation Metrics
#### Amazon Mechanical Turk (AMT) 
用于视觉评价， AMT是一个众包平台，将生成的结果发布到这个平台中，请人观察反馈。

#### Structural SIMilarity (SSIM) 
结构相似度， [百度百科](https://baike.baidu.com/item/SSIM/2091025?fr=aladdin), 结构相似性的范围为0到1。当两张图像一模一样时，SSIM的值等于1。

Code: [pytorch-ssim](https://github.com/Po-Hsun-Su/pytorch-ssim)

#### Inception Score (IS) 

Code: [inception-score-pytorch](https://github.com/sbarratt/inception-score-pytorch)  
讨论： [知乎](https://www.zhihu.com/question/297551781/answer/506852113)
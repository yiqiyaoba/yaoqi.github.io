---
layout: post
title: Semantic Image Synthesis via Adversarial Learning(ICCV2017)
mathjax: true
categories: Paper
tags: text2img,
keywords: text, img
description: Note of Semantic Image Synthesis via Adversarial Learning
mermaid: true
status: Writing
---

# 资料
> Paper: [Semantic Image Synthesis via Adversarial Learning](https://arxiv.org/pdf/1707.06873.pdf)  
> Source Code: [dong_iccv_2017](https://github.com/woozzu/dong_iccv_2017)  
> My Code: [Text2Img_Birds](https://github.com/huangtao36/Text2Img_Birds)


# Paper Note
## Dataset（Birds）
论文中使用到了 Oxford-102 flowers 和 Caltech-200 birds 数据集，这里只用 Birds 数据。

Download Dataset:  
> Caltech-200 birds: [images](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) and [captions](https://drive.google.com/file/d/0B0ywwgffWnLLLUc2WHYzM0Q2eWc/view?usp=sharing)  
> and The caption data is from [icml2016](https://github.com/reedscot/icml2016).  
> Caltech-200 birds 共包括 200类-11,788 张鸟的图片

分别下载 Image 和 Caption数据，解压后得到 CUB_200_2011 和 cub_icml 两个文件夹， 使用到的文件目录有：  
> Caltech200_birds/CUB_200_2011/images  # 包含200类的鸟的图片，以类别为二级目录  
> Caltech200_birds/cub_icml  # 对应每一个 image 的 caption 数据，.t7 文件  
> Caltech200_birds/cub_icml/trainvalclasses.txt   # 用于训练的类别目录

.t7 文件使用 **from torch.utils.serialization import load_lua** 直接加载： **load_lua('xxx.t7')** , 得到一个字典，字典包含： 'char', 'img', 'txt', 'word' 四个 key, 这里只用第一、二个: **char， img**. (其余未作了解)

> t7\_dic['char'].shape is [201, 10], 包含10句话，每句做多 201 个字符，   
> 字符对应： alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "    
> t7\_dic['img'] 是对应的图片的路径。  


使用下面的函数将其翻译为 sentences.

```python
alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
def _nums2chars(nums):
    chars = ''
    for num in nums:
        chars += alphabet[num - 1]
    return chars
```

Caption 经过 Dataloader 出来的数据流程： 

<img src="https://raw.githubusercontent.com/huangtao36/huangtao36.github.io/master/_posts/2019-03-11-SemanticImageSynthesis/assets/caption_dataloader.png" style="zoom:80%" /> 

## Text Embedding (Caption Encode)




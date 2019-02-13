---
layout: post
title: Text Encode Method in FashionGAN
mathjax: true
categories: Paper
tags: text
keywords: 
description:  FashionGAN text encode method
mermaid: true
status: Writing
---

> **起源**： 在复现 [FashionGAN](https://arxiv.org/abs/1710.07346)  的过程中使用到的一个文本编码的方法，现进行追溯。
> 
> FashionGAN中原文表示， Text Encoder 的方法来源于 [S. Reed et al.(ICMR, 2016)](https://arxiv.org/abs/1605.05396), FashionGAN 的官方代码中也给出了其 [Python Code](https://github.com/zhusz/ICCV17-fashionGAN/tree/master/language),

**本文解决如下几个问题**：   

- text encoder 的理论基础
- text encoder 是否如 FashionGAN 中原文描述一致

## ICMR2016 中的 Text Encoder 方法
原文中表述，这是一种 character-level text encoder， ﬁrst pre-train a deep convolutional recurrent text encoder on structured joint embedding of text captions with 1,024-dimensional GoogLeNet image embedings (Szegedy et al.  Going deeper with convolutions, 2015) as described in subsection 3.2.
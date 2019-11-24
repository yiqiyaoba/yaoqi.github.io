---
layout: post
title: Dynamic Few-Shot Visual Learning without Forgetting (CVPR2018)
mathjax: true
categories: Paper
tags: [Paper, Few-shot]
keywords: FSL, Few-shot
description: 

---

> The goal of this work is to devise a few-shot visual learning system that during test time it will be able to efﬁciently learn novel categories from only a few training data while at the same time it will not forget the initial categories on which it was trained (here called base categories).
>



### Summary

**Propose**: 

- To extend an object recognition system with an attention based few-shot classiﬁcation weight generator.
- To redesign the classiﬁer of a ConvNet model as the cosine similarity function // between feature representations and classiﬁcation weight vectors. (apart from unifying the recognition of both novel and base categories, it also leads to feature representations that generalize better on “unseen” categories. )

**实验数据集**：  

- Mini-ImageNet： 

**实验结果**：

- 1-shot: 56.20%
- 5-shot: 73.00%

**Other**: 

> - jupyter notebook： [Github-Code](https://github.com/HX-idiot/Dynamic-Few-Shot-Visual-Learning-without-Forgetting)  
>
> - 数据集： [Link](https://mega.nz/#!rx0wGQyS!96sFlAr6yyv-9QQPCm5OBFbOm4XSD0t-HlmGaT5GaiE)  
> - 论文： [Dynamic Few-Shot Visual Learning without Forgetting](https://arxiv.org/pdf/1804.09458.pdf)  
> - 源码： [Github-Code](https://github.com/gidariss/FewShotWithoutForgetting)
> - 相关博客：
>   -  [few-shot learning是什么](https://blog.csdn.net/xhw205/article/details/79491649 )

---

### Few-shot learning 的要求

- the learning of the novel categories needs to be fast （快速学习新事物）
- to not sacriﬁce any recognition accuracy on the initial categories that the ConvNet was trained on, i.e., to not “forget”  （不忘记旧事物）

### 本文提出两个新的技术

#### 1、Few-shot classiﬁcation-weight generator based on attention.

**传统的图像分类方法**： 先提取图像的高维特征，然后通过分类器计算属于每一个类别的概率（这个概率向量这里成为 “分类权重向量（classiﬁcation weight vectors） ”）.  

这里使用一个额外的组件 “few-shot classiﬁcation weight generator”， 在接受新的事物时（1-5个新类别），生成新的分类权重向量。主要特征是：通过将注意力机制纳入基本类别的分类权重向量上，从而显式地利用了过去获得的有关视觉世界的知识。

#### 2、Cosine-similarity based ConvNet recognition model.

xxx



### Methodology

定义一个 具有 $K_{base}$ 个基本类别的数据集：  
$$
D_{t r a i n}=\bigcup_{b=1}^{K_{b a s e}}\left\{x_{b, i}\right\}_{i=1}^{N_{b}}
$$
where $N_{b}$ is the number of training examples of the $b$ -th category and $x_{b, i}$ is its $i$ -th training example. 

本文的目标是： 使用此数据集作为唯一输入，既能够准确地识别基本类别，又能在不忘记基本类别的前提下，动态地经过少量样本学习来识别新的类别。

#### Model 总览

<img src="https://raw.githubusercontent.com/huangtao36/huangtao36.github.io/master/_posts/2019-11-24-FSL/assets/1574575285411.png" alt="1574575285411" style="zoom:50%;" />

##### ConvNet-based recognition model

- A **feature extractor** $F(.| \theta)$ (with learnable parameters  $\theta)$ that extracts a $d$ -dimensional feature vector $z=F(x | \theta) \in \mathbb{R}^{d}$ from an input image $x,$ 
- A **classifier** $C\left(. | W^{*}\right),$ where $W^{*}=\left\{w_{k}^{*} \in \mathbb{R}^{d}\right\}_{k=1}^{K}$ are a set of $K^{*}$ classification weight vectors - one  per object category, that takes as input the feature representation $z$ and returns a $K^{*}$ -dimensional vector with the probability classification scores $p=C\left(z | W^{*}\right)$ of the $K^{*}$ categories. 

(也就是传统的分类模型的两个模块)

We learn the $\theta$ parameters and the classification weight vectors of the base categories $W_{base}=\left\{w_{k}\right\}_{k=1}^{K_{base}}$ such that by setting $W^{*}=W_{base}$  the ConvNet model will be able to recognize the base object categories.

#####  Few-shot classiﬁcation weight generator

>  Meta-learning mechanism. 在测试时，接受 $K_{novel}$ 个新类别的少量数据作为输入。

$$
D_{n o v e l}=\bigcup_{n=1}^{K_{n o v e l}}\left\{x_{n, i}^{\prime}\right\}_{i=1}^{N_{n}^{\prime}}
$$

where $N_{n}^{\prime}$ is the number of training examples of the $n$ -th novel category and $x_{n, i}^{\prime}$ is its $i$ -th training example.

> - novel category  $$n \in\left[1, N_{\text {novel }}\right]$$
> - few-shot classiﬁcation weight generator  $G(., . . | \phi)$
> - input the feature vectors  $Z_{n}^{\prime}=\left\{z_{n, i}^{\prime}\right\}_{i=1}^{N_{n}^{\prime}}$
> - training examples $N_{n}^{\prime}$
> - $z_{n, i}^{\prime}=F\left(x_{n, i}^{\prime} | \theta\right)$ 
> - classiﬁcation weight vector  $w_{n}^{\prime}=G\left(Z_{n}^{\prime}, W_{b a s e} | \phi\right)$  

简单来说就是使用预训练好的特征提取器提取新类别图像的特征，然后将其扔进 few-shot classiﬁcation weight generator 训练， 得到 classiﬁcation weight vector， 得到 $W_{\text {novel}}=\left\{w_{n}^{\prime}\right\}_{n=1}^{K_{n o v e l}}$ (the classiﬁcation weight vectors of the novel categories inferred by the few-shot weight generator).  最后，在 $C\left(. | W^{*}\right)$ 中合并两个分类权重向量： $W^{*}=W_{b a s e} \cup W_{n o v e l}$ ， 使 ConvNet 可以同时分类出 base 和 novel 中的类。



#### Model 详细

##### Cosine-similarity based recognition model

关于怎么在测试时合并可变数量的新的类别。














---
layout: post
title: Dynamic Few-Shot Visual Learning without Forgetting
mathjax: true
categories: Paper
tags: [paper]
keywords: FSL, Few-shot
description: 

---

> CVPR2018-paper.   
>
> The human visual system has the remarkably ability to be able to effortlessly learn novel concepts from only a few examples. Mimicking the same behavior on machine learning vision systems is an interesting and very challenging research problem with many practical advantages on real world vision applications. In this context, the goal of our work is to devise a few-shot visual learning system that
> during test time it will be able to efﬁciently learn novel cat- egories from only a few training data while at the same time it will not forget the initial categories on which it was trained (here called base categories).



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

---








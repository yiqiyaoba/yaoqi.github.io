---
layout: post
title: Gradient Penalty
mathjax: true
categories: Knowledge
tags: [GAN]
keywords: gradient penalty, WGAN-GP
description: 

---

>  对WGAN进行改进，提出了一种替代WGAN判别器中权重剪枝的方法 

> 资料：
>
> - [https://blog.csdn.net/Jasminexjf/article/details/82686953](https://blog.csdn.net/Jasminexjf/article/details/82686953)  
> - 重点推荐： [https://zhuanlan.zhihu.com/p/25071913](https://zhuanlan.zhihu.com/p/25071913)
> - 重点推荐： [https://www.zhihu.com/question/52602529/answer/158727900](https://www.zhihu.com/question/52602529/answer/158727900 )

**GAN优化历程**： GAN——DCGAN——LSGAN——WGAN——WGAN-GP（Gradient Penalty） 

**WGAN 相比于 DCGAN**:

- 判别器最后一层去掉sigmoid
- 生成器和判别器的loss不取log
- 每次更新判别器的参数之后把它们的绝对值截断到不超过一个固定常数c
- 不要用基于动量的优化算法（包括momentum和Adam），推荐RMSProp，SGD也行

Paper: [Improved Training of Wasserstein GANs](https://arxiv.org/pdf/1704.00028.pdf). 研究者们发现 WGAN 失败的案例通常是由在 WGAN中 使用**权重剪枝**来对 critic 实施 Lipschitz 约束导致的。在 WGAN-GP 中，研究者们提出了一种替代权重剪枝实施 Lipschitz约束的方法：**惩罚 critic 对输入的梯度**。该方法收敛速度更快，并能够生成比权重剪枝的WGAN更高质量的样本。 

> 网络剪枝和共享用于降低网络复杂度和解决过拟合问题。  模型剪枝被认为是一种有效的模型压缩方法。 
>
> 清华&伯克利重新思考6大剪枝方法：  [RETHINKING THE VALUE OF NETWORK PRUNING](https://arxiv.org/pdf/1810.05270.pdf) 

 **[Improved Training of Wasserstein GANs](https://arxiv.org/pdf/1704.00028.pdf) 的工作：** 

> - 通过小数据集上的实验，概述了判别器中的权重剪枝是如何导致影响稳定性和性能的病态行为的。
> - 提出具有梯度惩罚的WGAN（WGAN with gradient penalty），从而避免同样的问题。
> - 展示该方法相比标准WGAN拥有更快的收敛速度，并能生成更高质量的样本。
> - 展示该方法如何提供稳定的GAN训练：几乎不需要超参数调参，成功训练多种针对图片生成和语言模型的GAN架构

关于 WGAN 与 WGAN-GP的推理，可见重点推荐的两个知乎笔记。
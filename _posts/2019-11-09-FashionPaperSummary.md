---
layout: post
title: Awesome Fashion AI
mathjax: true
categories: Summary
tags: [Summary, Paper]
keywords: Fashion
description: summary all paper about fashion
---

> 整理近年关于时尚图像的文章, 数据集等。

## 与文本有关的

- Be Your Own Prada: Fashion Synthesis with Structural Coherence

  [[paper]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_Be_Your_Own_ICCV_2017_paper.pdf) [[web]](http://mmlab.ie.cuhk.edu.hk/projects/FashionGAN/) [[code(torch)]](https://github.com/zhusz/ICCV17-fashionGAN)

  FashionGAN, 基于简单的文本描述改变人物着装的颜色、袖长、款式。提出两阶段的生成模型，不需要配对数据训练，同时发布 Fashion Synthesis [[link](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/FashionSynthesis.html)] 数据集.

- Language Guided Fashion Image Manipulation with Feature-wise Transformations

  [[paper](https://arxiv.org/pdf/1808.04000.pdf)]

  FiLMGAN, 基于文本描述改变人物服装

- Bilinear Representation for Language-based Image Editing Using Conditional Generative Adversarial Networks

  [[paper](https://arxiv.org/pdf/1903.07499.pdf)] [[code(pytorch)](https://github.com/vtddggg/BilinearGAN_for_LBIE)]

  BilinearGAN, 基于文本描述改变人物服装, 代码中同时包含有 FiLMGAN 的实现。
  
- Semantically Consistent Hierarchical Text to Fashion Image Synthesis with an enhanced-Attentional Generative Adversarial Network

    [[Paper](http://openaccess.thecvf.com/content_ICCVW_2019/papers/CVFAD/Ak_Semantically_Consistent_Hierarchical_Text_to_Fashion_Image_Synthesis_with_an_ICCVW_2019_paper.pdf)]

    文本直接生成时尚图像

## 虚拟试穿

- Fashion++: Minimal Edits for Outfit Improvement-ICCV2019

  [[paper](https://arxiv.org/pdf/1904.09261.pdf)] [[Supplementary ](http://openaccess.thecvf.com/content_ICCV_2019/supplemental/Hsiao_Fashion_Minimal_Edits_ICCV_2019_supplemental.pdf)] [[web](http://vision.cs.utexas.edu/projects/FashionPlus/)] [[code_pytorch](https://github.com/facebookresearch/FashionPlus)]

  时尚迁移，minimal adjustments to a full-body clothing outfit that will have maximal impact on its fashionability

- M2E-Try On Net: Fashion from Model to Everyone 

  [[paper](https://arxiv.org/pdf/1811.08599.pdf )]

  迁移两张人物图片上的服装

- SwapGAN: A Multistage Generative Approach for Person-to-Person Fashion Style Transfer 

  [[paper](http://jultika.oulu.fi/files/nbnfi-fe201902256190.pdf)]

  可实现基于文本描述编辑服装， 服装+人物图像的试穿， 人物+人物图像服装迁移

- SwapNet: Image Based Garment Transfer_ECCV2018 

  [[paper](https://eccv2018.org/openaccess/content_ECCV_2018/papers/Amit_Raj_SwapNet_Garment_Transfer_ECCV_2018_paper.pdf)] [[code_pytorch](https://github.com/andrewjong/SwapNet)]

  迁移两张人物图片上的服装

- Toward Characteristic-Preserving Image-Based Virtual Try-On Network_ECCV2018 

  [[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Bochao_Wang_Toward_Characteristic-Preserving_Image-based_ECCV_2018_paper.pdf)] [[code_pytorch](https://github.com/sergeywong/cp-vton)]

  CP-VTON, 服装+人物图像的试穿

- Towards Multi-pose Guided Virtual Try-on Network_ICCV2019 

  [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Dong_Towards_Multi-Pose_Guided_Virtual_Try-On_Network_ICCV_2019_paper.pdf)]

  多姿态的服装+人物图像的试穿

- VITON: An Image-based Virtual Try-on Network_CVPR2018

  [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Han_VITON_An_Image-Based_CVPR_2018_paper.pdf)] [[code](https://github.com/xthan/VITON)]

  服装+人物图像的试穿
  
- Generating High-Resolution Fashion Model Images Wearing Custom Outfits

    [[Paper](http://openaccess.thecvf.com/content_ICCVW_2019/papers/CVFAD/Yildirim_Generating_High-Resolution_Fashion_Model_Images_Wearing_Custom_Outfits_ICCVW_2019_paper.pdf)]

    多件服装综合生成人物时尚图像

## 时尚图像编辑

- Fashion Editing with Adversarial Parsing Learning

    [[paper](https://arxiv.org/pdf/1906.00884.pdf)]
    
- FiNet: Compatible and Diverse Fashion Image Inpainting_ICCV2019(oral) 

    [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Han_FiNet_Compatible_and_Diverse_Fashion_Image_Inpainting_ICCV_2019_paper.pdf)] [[supplementary](http://openaccess.thecvf.com/content_ICCV_2019/supplemental/Han_FiNet_Compatible_and_ICCV_2019_supplemental.pdf)]

- Attribute Manipulation Generative Adversarial Networks for Fashion Images

    [[Paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Ak_Attribute_Manipulation_Generative_Adversarial_Networks_for_Fashion_Images_ICCV_2019_paper.pdf)]

    基于标签改变服装属性（颜色，袖子等）

## 其他

- Fashion-Gen: The Generative Fashion Dataset and Challenge

   [[paper](https://arxiv.org/pdf/1806.08317.pdf)] [[code](https://github.com/ElementAI/fashiongen-challenge-template)]

   发布新的数据集 Fashion-Gen [[link]( https://fashion-gen.com/ )], 包含时尚人物图像、鞋子、包包等随身物品以及对应的文本描述。完全使用 [AttnGAN](http://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_AttnGAN_Fine-Grained_Text_CVPR_2018_paper.pdf) 做的生成模型代码 [[code_pytorch](https://github.com/menardai/FashionGenAttnGAN)]

- Fashion-AttGAN: Attribute-Aware Fashion Editing with Multi-Objective GAN

  [[paper](http://openaccess.thecvf.com/content_CVPRW_2019/papers/FFSS-USAD/Ping_Fashion-AttGAN_Attribute-Aware_Fashion_Editing_With_Multi-Objective_GAN_CVPRW_2019_paper.pdf)]

  纯服装图像作为输入，根据标签改变服装颜色与袖长
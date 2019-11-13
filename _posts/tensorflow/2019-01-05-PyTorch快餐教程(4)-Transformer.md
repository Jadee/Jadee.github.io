---
title: PyTorch快餐教程(4)-Transformer
date: 2019-01-05
categories: PyTorch
tags:
- PyTorch
---

# 前言

深度学习已经从热门技能向必备技能方向发展。然而，技术发展的道路并不是直线上升的，并不是说掌握了全连接网络、卷积网络和循环神经网络就可以暂时休息了。至少如果想做自然语言处理的话并非如此。

2017年，Google Brain的Ashish Vaswani等人发表了《Attention is all you need》的论文，提出只用Attention机制，不用RNN也不用CNN，就可以做到在WMT 2014英译德上当时的BLEU最高分28.4.

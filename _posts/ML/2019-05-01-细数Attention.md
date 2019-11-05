---
title: 细数Attention
date: 2019-05-01
categories:
- 机器学习
tags:
- DNN
---

# 前言

## 为什么写这篇文章？

曾被paper中各种各样的Attentioin搞得晕晕乎乎，零零散散整理过一些关于Attention的笔记。现在将这些笔记整合起来，是对知识的一次梳理回顾总结，不求深刻，但求浅显。希望能帮助对Attention有迷惑的同学。

<!-- more -->

## 文章都说些什么？

Attention自2015年被提出后，在NLP领域，图像领域遍地开花。Attention赋予模型区分辨别能力，从纷繁的信息中找到应当focus的重点。2017年self attention的出现，使得NLP领域对词句representation能力有了很大的提升，整个NLP领域开启了全面拥抱transformer的年代。

本文会主要从2个方面来介绍Attention。

**初识Attention**：主要扒一扒Attention的历史，然后给出一个通用的框架来回答一个终极问题：what is Attention？

**细数Attention**：以上文给出的通用框架视角来审视所有的Attention，在这个章节，你会和各种各样的Attention相遇相识相恋（global/local, soft/hard, Bagdanau attention, Luong attention, self-attention, multi-head attention , 以及它们的别名），了解它们之间的联系与差异。

# 初识Attention

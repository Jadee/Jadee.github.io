---
title: Embedding那些事：from word2vec to item2vec
date: 2019-03-05
categories:
- 机器学习
tags:
- DNN
---

# 前言

embedding向量大家可能都用过，但对于NLP领域涉足不深的个人来说，对embedding的认知和理解并不是很深刻。为了进一步理解并且能更好地运用embedding技术，便做了相关的调研，尝试将其汇总总结，理清一条完整的思路，并写下此文，希望对大家理解embedding有所帮助。

<!-- more -->

本文主要介绍了NLP领域embedding的发展，word2vec的经典模型，以及embedding技术推广到其他（推荐／搜索）领域的运用。实践证明，embedding技术对于工业场景来说有着很大的价值和应用前景。

# Word Embedding

自然语言作为一种非结构化的数据，很难被机器处理或学习。自然语言要被机器理解，第一步就需要将自然语言符号化表示。词的向量化表示作为一种很有效的方法，可以定量地度量词之间的关系，挖掘词之间的联系。那么，向量为什么能表示词呢？词向量如何生成呢？


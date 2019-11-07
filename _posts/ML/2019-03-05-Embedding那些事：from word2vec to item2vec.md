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

## one-hot representation

one-hot编码是最简单也是最容易理解的一种表示方式，类似于索引的方式。向量的维度为词库（词汇表）的大小，具体的词映射到索引位置，置为1，其余位置为0，生成的向量用来唯一表示词库中的某一个词。

举个例子：  
King、Queen、Man、Woman、Child 组成的词库V，|V|=5，则one-hot编码后的各个词的词向量为：
```
king  = (1,0,0,0,0)
queen = (0,1,0,0,0)
man   = (0,0,1,0,0)
woman = (0,0,0,1,0)
child = (0,0,0,0,1)
```
显然，这种表示方式会带来一些问题：

1）词汇表庞大带来的维度灾难  
2）数据过度稀疏，表达效率低

那么，能不能将词向量映射到较低维度的特征空间呢？

设想一下，有"Royalty"(王权),"Masculinity"(男子气?), "Femininity"(女人味?)和"Age"(年龄)4个维度特征，我们试着将词汇表V里的词用这4个维度来表示：
```
King  = (0.99, 0.99, 0.05, 0.7)
Queen = (0.99, 0.05, 0.93, 0.6)
man   = (0.05,0.95,0.1,0.6)
woman = (0.05,0.1,0.93,0.6)
child = (0.01,0.2,0.4,0.3,0.2)
```

## Dristributed representation

Distributed representation方法可以解决one-hot的问题，通过训练，将词映射成较低维向量。而大部分情况下词向量的各个维度并不能很好地解释，可以理解为隐含特征。有了词向量，我们便可以很好描述词，挖掘词之间的关系了。有一个有趣的研究表明：

$\overline{King} - \overline{Man} + \overline{Woman} = \overline{Queen}$

从直观含义上，不难理解。

那么，如何训练得到word embedding向量呢？  
一种很常见的方法，使用神经网络模型。常用的两种模型结构介绍如下。

### CBOW（Continuous Bag-of-Word Model）

用一句话描述CBOW模型的原理：用周围词预测中心词。利用中心词的预测结果，不断更新周围词的词向量。

模型的网络结构如下：

![avatar](https://img-blog.csdn.net/20171122185816199?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvZ2l0aHViXzM2MjM1MzQx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

V是词库大小，N为隐藏层维度，C是上下文词数量。

这里输入层是由one-hot编码的输入上下文 ${x_1，x_2，\ldots，x_C}$ 组成，最后输出层是也被one-hot编码的输出单词 $y$。被one-hot编码的输入向量通过一个$V * N$ 维的权重矩阵 $W$ 连接到隐藏层；隐藏层通过一个 $N * V$ 的权重矩阵 $W′$ 连接到输出层。

* **input**：context的one-hot向量 $(x_1，x_2，\ldots，x_C)$

* **input -> hidden**：context词的平均向量 x 权重矩阵

$$ h = \frac{1}{C} W * (\sum_{i = 1}^C x_i) $$ 

&emsp;&emsp;该输出就是输入向量的加权平均

* **hidden-> output**

&emsp;&emsp;1. 为每一个词库中的词计算得到一个分数

$$ u_j = v_{wj}^{'T} * h $$

&emsp;&emsp;&emsp;其中 $v_{wj}^{'T}$ 是输出矩阵 $W'$ 的第 $j$ 列。

&emsp;&emsp;2. 使用softmax多分类模型，计算出输出每一个词的概率 $y_i$：

$$ y_{c,j} = p(w_{y,j}|w_1，w_2，\ldots，w_c) = \frac{exp(u_j)}{\sum_{j' = 1}^{V} u_{j'}} $$ 

训练的目标是最大化实际输出词索引位置 $j\*$ 的条件概率 $y_{j\*}$，进一步得到模型的loss function，首先就是定义损失函数，这个损失函数就是给定输入上下文的输出单词的条件概率，一般都是取对数，如下所示：

$$ E = -logp(w_O | w_I) = - v_{wo}^T * h - log \sum_{j' = 1}^{V} exp(v_{wj'}^T * h)$$

使用梯度上升的方式更新参数，推导过程省略，得到输出权重矩阵 $W'$ 的更新规则：

$$ w'^{new} = w'^{old}_{ij} - \eta * {y_j - t_j} * h_i$$

同理权重 $W$ 的更新规则如下：

$$ w^{new} = w^{old}_{ij} - \eta * \frac{1}{C} * EH $$

### Skip-gram

skip-gram模型则与CBOW相反，是用中心词预测周围词。利用周围词的预测结果，不断更新调整中心词向量。

模型网络结构如下：

![avatar](https://pic4.zhimg.com/80/v2-6f1787c54370882fd1125d5defb937e7_hd.jpg)

* **input**：中心词的one-hot向量 $(x_1，x_2，\ldots，x_C)$

* **input -> hidden**

$$ h = x^T W = W_{k,.} := v_{wI} $$

* **hidden-> output**

&emsp;&emsp;C个多分类问题，共享一个参数矩阵

$$ p(w_{c,j} = w_{O,c} | w_{I}) = y_{c,j} = \frac{exp(u_{c,j})}{\sum_{j'=1}^V exp(u_{j'})} $$

&emsp;&emsp;计算得到每个词的分数：

$$ u_{c,j} = u_j = v_{wj}^{'T} * h，\quad for \quad c = 1，2，\cdots，C $$

此时训练目标为最大化c个输出中实际词所在索引位置的预测概率，loss function 如下：

$$ E = -log p(w_{O,1},w_{O,2},\cdots,w_{O,C}) = - log \prod_{c=1}^C \frac{exp(u_{c,j})}{\sum_{j'=1}^V exp(u_{j'})} $$

使用梯度上升的方式更新参数，推导过程省略，得到输出权重矩阵 $W'$ 的更新规则：

$$ w'^{new} = w'^{old}_{ij} - \eta * \sum_{c=1}^C (y_{c,j} - t_{c,j}) * h_i$$

同理权重 $W$ 的更新规则如下：

$$ w^{new} = w^{old}_{ij} - \eta * \sum_{j=1}^V \sum_{c=1}^C (y_{c,j} - t_{c,j}) * w'_{ij} * x_j $$

可以看出，不管是CBOW还是skip-gram，都要预测词库中所有词的概率，计算复杂度很高。此外，skip-gram预测的次数是要多于CBOW的，每个词在作为中心词的时候，都要进行C次的预测、调整。

借鉴他人的比喻来形象描述一下二者的区别：在skip-gram模型里，每个词在作为中心词的时候，相当于1个学生（中心词）和C个老师（上下文）。C个老师都会对学生进行“专业”训练，结果就是学生的能力会更强一些，但是这样对于学生而言每个老师都要进行一对一指导，肯定会使用更长的时间；而CBOW则是C个学生和1个老师，C个学生（上下文）都会从老师（中心词）这里学到知识，但是老师是“广播”知识，教给大家的知识是相同的。除此之外，学生还会从别的老师的课堂中和大家一起学到相同的知识（还会成为别的中心词的上下文），这样效率更高，速度更快。

## word2vec 

上面介绍的CBOW和skip-gram模型都是原始的模型结构，不计工程代价且没有做任何效率优化的工作。可以看出，为了更新词向量，对于每一条样本，都要预测词库中每一个词的概率，并利用预测误差更新参数和词向量。为了提高计算效率，一个直观的想法就是限制每次更新向量的数量。下面将介绍word2vec模型，主要通过两种方法对传统模型进行改进：一种比较优雅，利用树结构达到分层softmax；另一种是通过采样的方法。

word2vec模型对从input层到hidden层的映射关系进行了简化。在skip-gram中，$h = v_{wI}$，隐藏层输出为输入中心词向量；在CBOW中，$h = \frac{1}{C}\sum_{c=1}^C v_{w_c}$，隐藏层输出为上下文向量取均值。


### Hierarchical Softmax

**Huffman树**

Huffman树具有最短加权路径等良好性质。根据词库中词频可以构建一个Huffman树，这样词频越高的词，路径最短。

模型结构.

分层Softmax使用Huffman树代替隐藏层和输出层的神经元，将原output层 V 个词映射为叶子节点，这样每个词都可以从根节点通过某个特定路径到达叶子节点。即在每层进行决策的时候可以看作是一个二分类问题，即Hierarchical Softmax名字的由来。

词w作为输出词的概率定义为：

# Reference

[1] Distributed Representations of Words and Phrases and their Compositionality.pdf

[2] Efficient Estimation of Word Representations in Vector Space.pdf

[3] word2vec Parameter Learning Explained.pdf

[4] microsoft 2016 ITEM2VEC- NEURAL ITEM EMBEDDING FOR COLLABORATIVE FILTERING .pdf

[5] airbnb 2018 Real-time Personalization using Embeddings for Search Ranking at Airbnb.pdf

[6] [word2vec原理推导与代码分析](http://www.hankcs.com/nlp/word2vec.html#h3-5)

[7] NLP中的文本表示方法：<https://zhuanlan.zhihu.com/p/42310942>

[8] <https://yq.aliyun.com/articles/176894>

[9] <https://blog.csdn.net/weixin_41843918/article/details/90312339>

[10] <https://www.jianshu.com/p/1c73e01f9e5c>

[11] <https://www.jianshu.com/p/cdb93906607b>


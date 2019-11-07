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

## History 

Attention的发展可以粗暴的分为两个阶段。

2015-2017年，attention自提出后，基本成为NLP模型的标配，各种各样的花式attention铺天盖地。不仅在Machine Translation，在Text summarization，Text Comprehend（Q&A）, Text Classification也广泛应用。奠定基础的几篇文章如下：

* 2015年 ICLR 《Neural machine translation by jointly learning to align and translate》首次提出attention（基本算公认的首次提出），文章提出了最经典的Attention结构（additive attention 或者 又叫bahdanau attention）用于机器翻译, 并形象直观地展示了attention带来源语目标语的对齐效果，解释深度模型到底学到了什么。人类表示很服气~~

* 2015年 EMNLP 《Effective Approaches to Attention-based Neural Machine Translation》在基础attention上开始研究一些变化操作，尝试不同的score-function，不同的alignment-function。文章中使用的Attention（multiplicative attention 或者 又叫 Luong attention）结构也被广泛应用。

* 2015年 ICML 《Show, Attend and Tell: Neural Image Caption Generation with Visual Attention》是attention（提出hard/soft attention的概念）在image caption上的应用，故事圆满，符合直觉，人类再次表示很服气。 在上面几篇奠基之作之上，2016,2017年attention开枝散叶，无往不利。Hiearchical Attention，Attention over Attention，multi-step Attention……叫得上名的，叫不上名。

2017年-至今是属于transformer的时代。基于transformer强大的表示学习能力，NLP领域爆发了新一轮的活力，BERT、GPT领跑各项NLP任务效果。奠基之作无疑是：

* 2017年 NIPS《Attention is all you need》提出transformer的结构（涉及self-attention，multi-head attention）。基于transformer的网络可全部替代sequence-aligned的循环网络，实现RNN不能实现的并行化，并且使得长距离的语义依赖与表达更加准确（据说2019年的transformer-xl《Transformer-XL：Attentive Lanuage Models Beyond a fixed-length context》通过片段级循环机制结合位置编码策略可以捕获更长的依赖关系）。

## what is Attention ？

直奔主题，终极叩问 "what is attention?" 。这个章节，尝试以统一的抽象的框架来定义attention。如同先编写一个抽象类，后续章节涉及所有的attention都继承于这个抽象类。这里写了两个抽象类，一个叫alignment-based，一个叫memroy-based。

### alignment-based

如下图所示的model setting，输入c（context，有的论文写s），y（input，有的地方也写作h），输出z。图中英文表达原汁原味, 细品一下。

![avatar](/images/ml/ml-15.png)

我们细拆Attention Model，以经典的Bahdanau attention为例，看看抽象出来的三部曲：

* **score function**：度量环境向量与当前输入向量的相似性；找到当前环境下，应该focus哪些输入信息；

$$ \quad e_{ij} = a(c,y_i) = v_a^T tanh(W_a*c + U_a*y_i) $$

* **alignment function**：计算attention weight，通常都使用softmax进行归一化；

$$ \quad a_{ij} = \frac{exp(e_{ij})}{\sum_{k = 1}^{T_z} exp(e_{ik})}$$

* **generate context vector function**：根据attention weight，得到输出向量；

$$ \quad z_i = \sum_i \alpha_{ij} * y_i $$

下图更加直观地展示，这三个接口的位置：

![avatar](/images/ml/ml-16.png)

自此之后，要认清一个attention的详情，只需要搞清楚这三个部分，所有的变换都是在3个位置进行调整，当然变化最丰富的是score function。在后一个章节会详细对比不同种类的attention在这三个维度上的变换。

### memroy-based

另一种视角是QKV模型，假设输入为q，Memory中以（k，v）形式存储需要的上下文。感觉在Q&A任务中，这种设置比较合理，恩，transformer是采用的这种建模方式。k是question，v是answer，q是新来的question，看看历史memory中q和哪个k更相似，然后依葫芦画瓢，根据相似k对应的v，合成当前question的answer。

![avatar](/images/ml/ml-17.png)

在这种建模方式下，也分为三步：

* address memory（score function）：$e_i = a(q, k_i)$, 在memory中找相似；  
* normalize（alignment function）： $\alpha_i = softmax(e_i)$；  
* read content（gen context vector function）： $c = \sum_i a_i * v_i$；

其实还是没有逃出上文三部曲的框架。只是将input分裂成了 (k, v) pair。

后文都会以统一的三部曲建模方式（score function，alignment function，generate context vector function）来分析所有attention。

# Attention in Detail

在上文，我们high-level地了解了attention三部曲的建模方式，接下来要把所有Attention拉出来排排坐。

## Framework

如下图, 通常听到的一些attention，他们的差异其实主要体现在score-function层面，其次是体现在generate context vector function的层面。我们分别来看看，这些attention之间的差异与联系。

![avatar](/images/ml/ml-18.png)

### generate context vector function

**hard / soft attention** 是在文章《Show, Attend and Tell: Neural Image Caption Generation with Visual Attention》提出的概念，最直观的一种理解是，hard attention是一个随机采样，采样集合是输入向量的集合，采样的概率分布是alignment function产出的attention weight。因此hard attention的输出是某一个特定的输入向量。soft attention是一个带权求和的过程，求和集合是输入向量的集合，对应权重是alignment function产出的attention weight。hard / soft attention 中，soft attention是更常用的（后文提及的所有attention都在这个范畴），因为它可导，可直接嵌入到模型中进行训练，hard attention 文中suggests a Monte Carlo based sampling approximation of gradient。

### alignment function

在soft attention中，又划分了global/local attention（In this paper ：《Effective Approaches to Attention-based Neural Machine Translation》）。直观理解就是带权求和的集合不一样，global attention是所有输入向量作为加权集合，使用softmax作为alignment function，local是部分输入向量才能进入这个池子。为什么用local，背后逻辑是要减小噪音，进一步缩小重点关注区域。接下来的问题就是，怎么确定这个local范围？文中提了两个方案local-m和local-p。local-m基于的假设生硬简单，就直接pass了。local-p有一个预估操作，预计当前时刻应该关注输入序列（总长度为S）的什么位置pt（引入了两个参数向量，vp，wp），然后在alignment function中做了一点儿调整，在softmax算出来的attention wieght的基础上，加了一个以pt为中心的高斯分布来调整alignment的结果。作者最后阐述local-p + general（score-function 参考上图中multiplicative attention中的general版本）的方式效果是最好的。但从global/local 视角的分类来看，更常用的依然还是global attention，因为复杂化的local attention带来的效果增益感觉并不大。

![avatar](/images/ml/ml-19.png)

### score-function

如何生成输出向量，有上面提及的那些变换。接下来是变化更加丰富的score function。最为常用的score function有上文图中的那几种（基本全乎了吧~）。其实本质就是度量两个向量的相似度。如果两个向量在同一个空间，那么可以使用dot点乘方式（或者scaled dot product，scaled背后的原因是为了减小数值，softmax的梯度大一些，学得更快一些），简单好使。如果不在同一个空间，需要一些变换（在一个空间也可以变换），additive 对输入分别进行线性变换后然后相加，multiplicative 是直接通过矩阵乘法来变换（你是不是也曾迷惑过为什么attention要叫做additive和multiplicative attention？）。

后文我们将介绍几个具有代表性的attention，通过具体的attention example来进一步理解。以及一些花样attention，看大家都怎样变着法儿用attention。

## Bahdanau Attention & Luong Attention

在对比之中，认知更清晰，一图表达所有。这两个Attention就是整个Attention的奠基之作。Tensorflow中实现了这两种Attention的API。

![avatar](/images/ml/ml-20.png)

## Self Attention & Multi-head Attention

### why self attention ?

有很多文章写self-attention，但是写why self-attention的并不多。所以打算多花点笔墨来写why。

RNN的长距离依赖比较tricky：RNN 很强大（可以作为encoder对长度任意的序列进行特征抽取，基于特征抽取的能力可以胜任分类任务，另一方面可以作为Generators学习Language Model），其实核心就是长距离依赖（gate architectures - 线性操作让信息可以保持并流动，并选择性地让信息通过），可以对长度任意的序列进行表达，但是这种方式还是比较tricky。并且这种序列建模方式，无法对具有层次结构的信息进行很好的表达。
RNN由于递归的本质，导致无法并行。

![avatar](/images/ml/ml-21.png)

CNN在NLP中扮演了n-gram的detector角色，在层内可以并行。CNN works well，基于的假设是局部信息相互依赖。CNN具有Hierarchical Receptive Field，使得任意任意两个位置之间的长度距离是对数级别的。

![avatar](/images/ml/ml-22.png)

所以有没有一种方法，能够做到既能又能还能？

* 相对于CNN，要constant path length 不要 logarithmic path length , 要 variable-sized perceptive field，不要固定size的perceptive field；  
* 相对于RNN，考虑长距离依赖，还要可以并行！

这就是self attention。下图可以看到self-attention和convolution有点儿神似，它摒弃了CNN的局部假设，想要寻找长距离的关联依赖。看下图就可以理解self-attention的这几个特点：

* constant path length & variable-sized perceptive field ： 任意两个位置（特指远距离）的关联不再需要通过 Hierarchical perceptive field的方式，它的perceptive field是整个句子，所以任意两个位置建立关联是常数时间内的。  
* parallelize : 没有了递归的限制，就像CNN一样可以在每一层内实现并行。

![avatar](/images/ml/ml-23.png)

self-attention借鉴CNN中multi-kernel的思想，进一步进化成为Multi-Head attention。每一个不同的head使用不同的线性变换，学习不同的relationship。

### what is self-attention？

已经有很多很好的文章介绍transformer以及self-attention，以及内部细节，不打算当搬运工。这里贴个链接：目前见过讲得最通俗易懂的文章：[transformer详细介绍](https://jalammar.github.io/illustrated-transformer/?spm=ata.13261165.0.0.245448aaVpSnpm)。下图是完整版本的multi-head attention的示例图（引用自上述链接中）。这是基于上文中提及了QKV的memory-based的建模方式。需要说明的几个点：

1. QKV都是对输入x的线性映射。  
2. score-function使用scaled-dot product。  
3. multihead的方式将多个head的输出z，进行concat后，通过线性变换得到最后的输出z。

![avatar](/images/ml/ml-24.png)

transformer框架中self-attention本身是一个很大的创新，另一个有意思的是 three ways of attention的设计。attention weight一列以英译中，encoder输入machine learning，decoder输入机器学习。

1. Encoder self-attention：Encoder阶段捕获当前word和其他输入词的关联；  
2. MaskedDecoder self-attention ：Decoder阶段捕获当前word与已经看到的解码词之间的关联，从矩阵上直观来看就是一个带有mask的三角矩阵；  
3. Encoder-Decoder Attention：就是将Decoder和Encoder输入建立联系，和之前那些普通Attention一样；

![avatar](/images/ml/ml-25.png)

transformer中除了上诉提及的东西，还有positional encoding，residuals这些小而美的东西。在复杂度方面在原文中也与RNN-CNN进行了对比。

### 花样Attention

下面简要介绍几种花样的attention：

RNN对序列建模，但是缺乏层次信息。而语言本身是具有层次结构，短语组成句子，句子组成篇章。因此研究者十分希望把语言中的这些层次结构在模型中得以体现，Hierarchical的方式就出现了。《Hierarchical Attention Networks for Document Classification》，从word attention到sentence attention，如下图一。

在匹配或者检索任务中（如Q&A，IR），要衡量query，doc相似度，这时候attention的方法中，query和doc就互为对方的cotext，query对doc算一次attention，doc对query算一次attention，《Attention-over-Attention Neural Networks for Reading Comprehension 》，如下图二。

上文介绍why self-attention时已经提及了RNN和CNN的一些优点和问题，几乎和transformer同时，facebook发表了《Convolutional Sequence to Sequence Learning》，同样地想借用CNN的优点来补足RNN不能并行的弱点，用CNN结合attention来对序列进行建模，如下图三。

随着transformer的爆红，围绕transformer的花边，出现了weighted-transformer 《Weighted Transformer Network For Machine Translation》。 今年出现了transformer-xl 《Transformer-xl ： attentive language models beyond a fixed-length context》，如下图四， 想达到对任意长度的输入进行特征抽取，而不是transformer切成segment的定长输入。

![avatar](/images/ml/ml-26.png)

# 总结

**Why Attention Works？**

从上面的建模，我们可以大致感受到Attention的思路简单，四个字“带权求和”就可以高度概括，大道至简。做个不太恰当的类比，人类学习一门新语言基本经历四个阶段：死记硬背（通过阅读背诵学习语法练习语感）->提纲挈领（简单对话靠听懂句子中的关键词汇准确理解核心意思）->融会贯通（复杂对话懂得上下文指代、语言背后的联系，具备了举一反三的学习能力）->登峰造极（沉浸地大量练习）。 这也如同attention的发展脉络，RNN时代是死记硬背的时期，attention的模型学会了提纲挈领，进化到transformer，融汇贯通，具备优秀的表达学习能力，再到GPT、BERT，通过多任务大规模学习积累实战经验，战斗力爆棚。

要回答为什么attention这么优秀？是因为它让模型开窍了，懂得了提纲挈领，学会了融会贯通。

那又是如何开窍的？是因为它懂得了"context is everything"。

1. 在语言模型中：语言模型（language model）是整个NLP领域的基础，语言模型的精准程度基本上直接掌握所有NLP任务效果的命脉。而context又掌握着语言模型的命脉，语义不孤立，在特定context下展示特定的一面，模型如果可以学习到这些知识，就可以达到见人说人话，见鬼说鬼话的理想状态。在语义表达上能把context用好的都是成功的典范（参考：word2vec靠学习word及其context发家，ELMo-deep contextualized word representations， BERT从句子中抠掉一个词用上下文去预测这个词, transformer-xl 较 transformer使用更全面的context信息，XLNet一大重要贡献也是研究如何使用上下文信息来训练语言模型）。

2. 在其他领域中：Attention是把context用好的典范之一。Attention背后本质的思想就是：在不同的context下，focusing不同的信息。这本来就是一个普适的准则。所以Attention可以用到所有类似需求的地方，不仅仅是NLP，图像，就看你对context如何定义。在很多的应用场景，attention-layer肩负起了部分feature-selection,featue-representation的责任。举个栗子，transfer learning with Domain-aware attention network for item recommemdation in e-commerce 中提及：不同场景的用户的行为有不同的偏好（场景是context，价格，品牌是不同的信息），天猫用户对品牌看重，亲淘用户focus价格，可以通过attention-layer学习到不同context下，用户的Attention在哪里。在ctr预估中，[Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/abs/1706.06978?spm=ata.13261165.0.0.245448aaxZlLUF&file=1706.06978) 出发点类似。 在推荐场景中，文章Feature Aware Multi-Head Attention在手淘猜你喜欢排序模型中的应用 。这些都是attention在业务场景落地的参考。

# Reference

[1] Bahdanau D, Cho K, Bengio Y. Neural machine translation by jointly learning to align and translate[J]. arXiv preprint arXiv:1409.0473, 2014.  
[2] Xu K, Ba J, Kiros R, et al. Show, attend and tell: Neural image caption generation with visual attention[J]. arXiv preprint arXiv:1502.03044, 2015.  
[3] Luong M T, Pham H, Manning C D. Effective approaches to attention-based neural machine translation[J]. arXiv preprint arXiv:1508.04025, 2015.  
[4] Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need[C]//Advances in neural information processing systems. 2017: 5998-6008.  
[5] Dai Z, Yang Z, Yang Y, et al. Transformer-xl: Attentive language models beyond a fixed-length context[J]. arXiv preprint arXiv:1901.02860, 2019.  
[6] Yang Z, Yang D, Dyer C, et al. Hierarchical attention networks for document classification[C]//Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. 2016: 1480-1489.  
[7] Cui Y, Chen Z, Wei S, et al. Attention-over-attention neural networks for reading comprehension[J]. arXiv preprint arXiv:1607.04423, 2016.  
[8] Gehring J, Auli M, Grangier D, et al. Convolutional sequence to sequence learning[C]//Proceedings of the 34th International Conference on Machine Learning-Volume 70. JMLR. org, 2017: 1243-1252.  
[9] <https://github.com/kimiyoung/transformer-xl>  
[10]<https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html>  
[11]<https://jalammar.github.io/illustrated-transformer/>  
[12]<https://www.jianshu.com/p/b1030350aadb>  
[13]<https://arxiv.org/abs/1706.06978>

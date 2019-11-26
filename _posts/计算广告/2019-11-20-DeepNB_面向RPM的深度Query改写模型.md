---
title: DeepNB_面向RPM的深度Query改写模型
date: 2019-11-20
categories: 计算广告
tags:
- 计算广告
- 广告触发
---

# 业务背景

在搜索营销系统中，广告是被挂载在竞价词（bidword）下的。用户查询词（query）作为倒排索引key去检索广告，被召回的广告再经过多轮排序后投放给用户。广告主则主要通过为广告购买bidword参与流量分配。这种方式是“精确匹配”。然而，由于用户搜索query的多样性，广告主列举相关的bidword是非常耗时耗力的。为此，广告平台提供了改写（Query Rewrite）算法，实现将用户query自动改写到相关的bidword上，此为“模糊匹配”。QR减轻了广告主的营销负担，同时也能提高平台变现效率。

<!-- more -->

一个典型的搜索广告系统如下图所示。可以看到，QR模块处于广告系统的最前端，它决定了哪些广告能够被召回，因此对最终效果有着重要影响。本文将探讨在搜索直通车业务中如何设计面向RPM的深度QR算法。

![avatar](/images/trigger/trigger-8.png)

项目论文：AAAI 2020 <https://arxiv.org/abs/1910.12527>

# 技术演进

改写算法主要分为3个阶段:

* 第一代是面向相关性的改写算法，强调query与bidword的语义相关性，例如经典的基于图挖掘的SimRank++[1]。这一代改写算法的缺陷是未能考量bidword侧的流量分配机制，广告营收潜力难以释放。

* 第二代在关注相关性的同时面向了RPM建模，将改写建模为图覆盖问题（Graph Covering Problem），兼顾了用户体验和平台营收。例如淘宝搜索直通车的“Nodebidding”算法，Yahoo也有类似的工作[2]。简单来说，Nodebidding首先确定query下优质广告集合，然后再反过来选取bidword，使得bidword挂载的广告对优质广告集合实现近似最大覆盖。Nodebidding经过了多次迭代，取得了非常优异的效果。但继续沿着这条路进行效果提升越来越困难，主要遇到了以下几个问题：

  * Nodebidding将query看作一个不可分割的graph node，这是Nodebidding的一大特色，同时也有所限制了对query进行更加深入的理解。例如，“大码女装”和“胖mm衣服”从人类语义上讲是很接近的，而Nodebidding则是将两者作为独立的实体对待，难以将从其中一个身上学到的信息迁移到另一个上面。  
  * Nodebidding本质上是基于统计的记忆模型，所以要在长尾流量上面挖掘出置信度高的改写关系是相对困难的。而从图2可见，直通车场景下日PV不足50个的query占比高达90%，稀疏的数据影响了算法在长尾上的精度；而大盘有30%的PV来自于这些长尾query，具有较高的商业价值；同时，长尾query目前覆盖了全库67.2%的广告，优化长尾query的表现对效果具有可观的收益。
  
![avatar](/images/trigger/trigger-9.png)

  * Nodebidding对流量的切分粒度是搜索词粒度，而随着个性化算法的发展，我们迫切需要对流量的进行更加精细化的耕作。

为了解决上述问题，在Nodebidding的框架下尝试了多种改进，如：基于协同过滤的query扩展、基于term match的长尾改写、基于前置点击item的个性化改写等等。这些算法在一定程度上缓解了上述问题，并在线上取得了不错的效果，但并没有优雅彻底地解决上面三个挑战。随着在这一领域的实践和思考的不断深入，因此提出了新一代的改写模型——DeepNB来更好的应对这些挑战。

# 新一代改写模型——DeepNB

## 模型概述

DeepNB通过深度网络将query和bidword映射为同一隐式空间中的向量，将bidword的改写概率转化为两个向量之间的距离计算。在对query进行embedding时，DeepNB不再将其局限为搜索词，而是引入了丰富的特征维度，并纳入了user的信息，提升了模型对query理解的泛化能力和精细程度，改善了模型在长尾query上的表现。在bidword侧，DeepNB设计了面向RPM的样本构造方法，使得高RPM潜力的bidword能以更大的概率被改写出来，近似实现对优质广告集的最大图覆盖。下面几节将做详细介绍。

![avatar](/images/trigger/trigger-10.png)

### Query侧的网络结构

不同于后续排序模型，QR面对的是上亿量级的bidword库，并需要在1~2毫秒的响应时间内确定一个小规模的改写集，这对平衡effective和effciency提出了很高的要求。为了应对这一严苛挑战，DeepNB在网络结构设计上力求精简高效。

DeepNB的query侧网络有三类输入特征，分别是：query分词后的term特征、query归一化后的id、用户profile。query分词后，首先对term list进行embedding，然后将embedding向量通过Attention Based CNN网络。这里试图利用CNN网络在捕捉局部组合特征上的优秀能力来实现n-gram效果。在pooling层加入Attention机制，因为我们发现在淘宝搜索场景下，用户的搜索词大部分是“属性+商品名”这样的组合，属性词能精确地表达了用户个性需求，而在圈定类目后商品名所携带的信息就大为减少，所以在改写时需要通过注意力机制来给予不同的词以不同的权重。

![avatar](/images/trigger/trigger-11.png)

我们同时加入了对top query进行id化之后的embeddding向量，以此增强模型在头部流量上的记忆能力。

用户profile特征让模型挖掘出隐含属性（如图中：“女性”），使之具备个性化改写的能力。

截止目前，Query Term和Query id两类特征加入后，取得了显著效果。User profile特征也将引入。

三类特征分别得到64维向量，然后经过Weighted Sum得到合并的向量表示。再经过三层全连接层，最后得到16维的query向量。

### Bidword侧的正负例构造

DeepNB在bidword侧没有设计泛化特征，而是在id化后直接将其映射为16维向量。这里的原因是：DeepNB认为bidword本质上只是一个广告集合的表征，它的效果表现完全取决于被挂载上去的广告，而跟它自身的诸如语义分词之类的特征没有关系。

正负bidword的构造依赖于学习目标的设定，如果模型希望改写出点击率效果好的bidword，那么正例就是那些挂载了高点击率广告的bidword；而如果希望改写模型能提高平台RPM收益，就需要将挂载了高营收潜力广告的bidword置为正例。

由于query与bidword之间没有天然的label，所以我们在构建正负例时采取将广告label传播给bidword的方式。对于每一次广告点击，该广告所有购买的bidword都被“激活”，然后可以有三种样本构造方式：

* 第一种是将激活的bidword全部作为正例。这种方式简单直观，但会导致训练样本的急剧膨胀。假如每天的点击数据量在5亿，每个广告平均购买bidword有200个，那么，一天的训练样本就有千亿级别。

![avatar](/images/trigger/trigger-12.png)

* 第二种是只使用当次召回的bidword作为正例。在线上环境中，广告的每一次展现都是被某个bidword召回的，所以我们可以只选择当次bidword作为正例。这种方式克服了第一种方式的样本爆炸的缺陷，缺点是可能会使算法陷入自循环，加剧马太效应。

![avatar](/images/trigger/trigger-13.png)

* 第三种方式是在激活的bidword中根据某种采样算法挑选有限几个bidword作为正例。这种方法大大减少了样本量级，同时缓解了马太效应。采样算法需要设计合理，不然会严重影响效果。

![avatar](/images/trigger/trigger-14.png)

DeepNB采用了第三种方式，并设计了面向RPM的采样算法。

设竞价词 $b_i$ 下挂载的广告数量为 $n_{b_i}$，广告 $a$ 在 $b_i$ 上的出价为 $price(b_i \| a)$，那么对每条点击样本<q, a>，竞价词 $b_i$ 被采样为正例的得分为：

$$ score(b_i |a, q) = \frac{price(b_i | a)}{log(n_{b_i} + 1)} \tag{1}$$

然后我们对广告 $a$ 购买的所有 bidword 的 score 进行归一化，得到最终的正例 label 传播概率：

$$ p(b_i |a, q) = \frac{score(b_i |a, q)}{\sum_{b_k \in B(A)} score(b_k |a, q) } \tag{2}$$

我们根据正例label传播概率采样 $m(10 \\, \geq m \geq \\, 1)$ 个bidword作为正例，将每条点击样本扩展m倍。

可以证明，按照上述采样方法构造的样本能使得query改写出bidword的概率（也就是向量距离）与RPM是正相关的。这里做一简要说明。

对于 $<q, b_i>$ 的改写端RPM，可以有如下推导：

$$ \begin{align} QR\_RPM(b_i, q) & = \frac{revenue(q, b_i)}{request(q)} \\
& = \frac{\sum_{j=1}^{n_{b_i}} \: price(b_i | a_j) * click(a_j|q) }{  request(q) } \\
& = \sum_{j=1}^{n_{b_i}} \: price(b_i | a_j) * \frac{click(a_j|q)}{request(q) } \\
& = u(q) * \sum_{j=1}^{n_{b_i}} \: price(b_i | a_j) * \frac{click(a_j|q)}{click(q) }
\end{align}
$$

上式中：$click(q) = request(q) * pvr(q) * ctr(q)$。

对于最后的 $q -> b_i$ 的改写概率，有如下推导：

$$ \begin{align} p(b_i | q) & = \sum_{j=1}^{n_{b_i}} \: p(b_i | a_j, q) * p(a_j | q)  \\
& \infty \sum_{j=1}^{n_{b_i}} \: price(b_i | a_j) * \frac{click(a_j|q)}{click(q) } \\
& = \frac{1}{u(q)} * QR\_RPM(b_i, q) 
\end{align}
$$

也就是得到了：

$$ p(b_i | q) \: \infty \: QR\_RPM(b_i, q) $$ 

这个式子的物理含义是：q 被改写为 $b_i$ 的概率正比于 $b_i$ 在 q 下的期望RPM，反比于 $b_i$ 挂载的广告数量。之所以加上广告数量因子，是因为我们希望在相同RPM潜力的情况下，让模型更倾向于选择不那么热门的bidword，使得广告竞争更加均衡。

上面（1）式的分子项还可以乘上q与b的语义相似度，以增强改写的语义相关性。

对于负样本的构造，我们借鉴了“课程学习(curriculum learning)”的思想，给模型喂给不同分类难度的负样本。以类目是否相同为easy negative的判别标准，以 $q = 1 - p(b_i \| a, q)$ 值作为hard negative的判断指标。在训练模型时，先学习easy negative，然后再使用hard negative。

![avatar](/images/trigger/trigger-15.png)

下面以一个示例来说明DeepNB正负bidword的采样算法。

![avatar](/images/trigger/trigger-16.png)

## 模型训练

模型训练时，DeepNB首先计算query分别与正例bidword和负例bidword之间的向量cos距离，然后最大化query与正例之间的cos距离。损失函数为：

$$ L(\Lambda) = -log \prod_{q, b^{+}} \: \frac{ exp(\gamma cos(q, b^{+}))  }{ \sum exp(\gamma cos(q, b^{+}))  } $$

根据在样本构造时的分析，DeepNB模型将给更具RPM潜力的bidword赋予更高的预测权重。因此，如果我们贪心地选取距离相近的bidword，那么就能近似实现对优质广告集合的最大图覆盖。

## 在线服务

前文提到QR阶段对响应时间的要求很高，为了更好地适应这个需求，我们在在线服务阶段将流量拆分为头部和长尾两类，分别进行处理。

对于头部300W的query，我们预先调用模型对query和bidword进行离线inference，然后进行向量距离计算，得到query距离相近的bidword作为改写结果。这些改写结果将被存储在Tair中。在线服务时，直接使用query查找出对应的改写结果。提前cache头部query的改写结果可以显著缩短QR的平均响应时间。

对于非头部300W的query，我们需要进行实时向量inference和向量距离计算。在线请求到来时，首先依次调用特征生成（Feature Generator）和特征转换（Feature Transformer）模块，得到query的可输入模型的特征。然后调用online model inference模块（Prophet）计算出query的向量。接着用query向量去向量近邻检索模块（ANN）获取近邻bidword，并依据向量距离进行排序，截取topN。最后由广告倒排检索模块召回广告给后续的排序阶段。

![avatar](/images/trigger/trigger-17.png)

# 参考文献

1、Simrank++: Query Rewriting through Link Analysis of the Click Graph.  
2、Optimizing Query Rewrites for Keyword-Based Advertising.  
3、Convolutional Neural Networks for Sentence Classification. <https://arxiv.org/pdf/1408.5882.pdf>  
4、An Introduction to Deep Learning for Natural Language Processing.  
5、ABCNN Attention-Based Convolutional Neural Network for Modeling Sentence Pairs.  
6、Deep Learning in Natural Language Processing.

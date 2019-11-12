---
title: GitHub-各大数据竞赛 Top 解决方案开源汇总
date: 2019-05-14
categories:
- Github
tags:
- github
---

# 背景

现在，越来越多的企业、高校以及学术组织机构通过举办各种类型的数据竞赛来「物色」数据科学领域的优秀人才，并借此激励他们为某一数据领域或应用场景找到具有突破性意义的方案，也为之后的数据研究者留下有价值的经验。

<!-- more -->

[Smilexuhc](https://github.com/Smilexuhc?spm=ata.13261165.0.0.5eb34f452U2HdZ) 在 GitHub 社区对各大数据竞赛名列前茅的解决方案进行了整理，包括纯数据竞赛、自然语言处理（NLP）领域数据赛事的 Top 解决方案。对这些赛事感兴趣的小伙伴可以一起来看一下这篇干货满满的汇总贴。

# 数据竞赛

## 2018 科大讯飞 AI 营销算法大赛

本次大赛要求参赛者基于提供的讯飞 AI 营销云的海量广告投放数据，通过人工智能技术构建来预测模型预估用户的广告点击概率。比赛提供了 5 类数据，包括基础广告投放数据、广告素材信息、媒体信息、用户信息和上下文信息，总共为 1001650 初赛数据 和 1998350 条复赛数据（复赛训练数据为：初赛数据+复赛数据）。

```
Rank1：<https://zhuanlan.zhihu.com/p/47807544>
```

## 2018 IJCAI 阿里妈妈搜索广告转化预测

本次比赛要求参赛者以阿里电商广告为研究对象，基于提供的淘宝平台的海量真实交易数据，通过人工智能技术构建来预测模型预估用户的购买意向。本次比赛为参赛者提供了 5 类数据，包括基础数据、广告商品信息、用户信息、上下文信息和店铺信息。用于初赛的数据包含了若干天的样本；最后一天的数据用于结果评测，对选手不公布；其余日期的数据作为训练数据，提供给参赛选手。

```
* Rank1：<https://github.com/plantsgo/ijcai-2018> 

* Rank2：<https://github.com/YouChouNoBB/ijcai-18-top2-single-mole-solution> 

&emsp;&emsp;<https://blog.csdn.net/Bryan__/article/details/80600189> 

* Rank3: <https://github.com/luoda888/2018-IJCAI-top3> 

* Rank8: <https://github.com/fanfanda/ijcai_2018> 

* Rank8: <https://github.com/Gene20/IJCAI-18> 

* Rank9（第一赛季）：<https://github.com/yuxiaowww/IJCAI-18-TIANCHI> 
```

## 2018 腾讯广告算法大赛

本次算法大赛的题目源自一个基于真实业务场景的广告技术产品——腾讯社交广告 Lookalike 相似人群拓展。本题目要求参赛者基于提供的几百个种子人群、海量候选人群对应的用户特征，以及种子人群对应的广告特征，构建算法准确标定测试集中的用户是否属于相应的种子包。

出于业务数据安全保证的考虑，比赛所提供的所有数据均为脱敏处理后的数据。整个数据集分为训练集和测试集：训练集中标定了人群中属于种子包的用户与不属于种子包的用户（即正负样本），测试集将检测参赛选手的算法能否准确标定测试集中的用户是否属于相应的种子包，训练集和测试集所对应的种子包完全一致。初赛和复赛所提供的种子包除量级有所不同外，其他的设置均相同。

```
* Rank3: <https://github.com/DiligentPanda/Tencent_Ads_Algo_2018> 

* rank6: <https://github.com/nzc/tencent-contest> 

* Rank7: <https://github.com/guoday/Tencent2018_Lookalike_Rank7th> 

* Rank9: <https://github.com/ouwenjie03/tencent-ad-game> 

* Rank10: <https://github.com/keyunluo/Tencent2018_Lookalike_Rank10th> 

* rank10（初赛）: <https://github.com/ShawnyXiao/2018-Tencent-Lookalike> 

* Rank11: <https://github.com/liupengsay/2018-Tencent-social-advertising-algorithm-contest>  
&emsp;&emsp;<https://my.oschina.net/xtzggbmkk/blog/1865680> 
```

## 2018 高校大数据挑战赛—快手活跃用户预测 

本次大赛要求参赛者基于脱敏和采样后的数据信息，预测未来一段时间活跃的用户。参赛队伍需要设计相应的算法进行数据分析和处理，比赛结果按照指定的评价指标使用在线评测数据进行评测和排名。大赛提供的数据为脱敏和采样后用户行为数据，日期信息进行统一编号，第一天编号为 01，第二天为 02，以此类推，所有文件中列使用 tab 分割。

```
* Rank1：<https://github.com/drop-out/RNN-Active-User-Forecast>  
&emsp;&emsp;<https://zhuanlan.zhihu.com/p/42622063> 

* Rank4: <https://github.com/chantcalf/2018-Rank4-> 

* Rank13(初赛 a 榜 rank2；b 榜 rank5)：<https://github.com/luoda888/2018-KUAISHOU-TSINGHUA-Top13-Solutions> 

* Rank15: <https://github.com/sunwantong/Kuaishou-Active-User> 

* Rank20: <https://github.com/bigzhao/Kuaishou_2018_rank20th> 
```

## 2018JDATA 用户购买时间预测

本次大赛要求参赛者基于给定的近 3 个月购买过目标商品的用户以及他们在前一年的浏览、购买、评价等数据信息，自行设计数据处理相关操作、训练模型，从而预测未来 1 个月内最有可能购买目标品类的用户，并预测他们在考察时间段内的首次购买日期。数据主要包括用户基本信息、SKU 基本信息、用户行为信息、用户下单信息及评价信息。

```
* Rank9：<https://zhuanlan.zhihu.com/p/45141799>
```

## 2018 DF 风机叶片开裂预警

本次大赛要求参赛者基于风机 SCADA 实时数据，通过机器学习、深度学习、统计分析等方法建立叶片开裂早期故障检测模型，对叶片开裂故障进行提前告警。比赛提供的数据集包括训练集和测试集：训练集一共有 25 类风机共 4 万个样本，测试集没有风机编号，共 8 万个样本。

```
* Rank2：<https://github.com/SY575/DF-Early-warning-of-the-wind-power-system>
```

## 2018 DF 光伏发电量预测

本次大赛要求参赛者在分析光伏发电原理的基础上，论证辐照度、光伏板工作温度等影响光伏输出功率的因素，通过实时监测的光伏板运行状态参数和气象参数建立预测模型，预估光伏电站瞬时发电量，并根据光伏电站 DCS 系统提供的实际发电量数据进行对比分析，验证模型的实际应用价值。

比赛提供训练集 9000 个点，测试集 8000 个，包括光伏板运行状态参数（太阳能电池板背板温度、其组成的光伏阵列的电压和电流）和气象参数（太阳能辐照度、环境温湿度、风速、风向等）。

```
* Rank1：<https://zhuanlan.zhihu.com/p/44755488?utm_source=qq&utm_medium=social&utm_oi=623925402599559168>  
&emsp;&emsp;这一方案也可查看微信文章：[《XGBoost+LightGBM+LSTM:一次机器学习比赛中的高分模型方案》](https://mp.weixin.qq.com/s/Yix0xVp2SiqaAcuS6Q049g)
```

## AI 全球挑战者大赛—违约用户风险预测

本次大赛要求参赛者基于马上金融平台提供的近 7 万贷款用户的基本身份信息、消费行为、银行还款等数据信息，建立准确的风险控制模型，来预测用户是否会逾期还款。

```
* Rank1：<https://github.com/chenkkkk/User-loan-risk-prediction>
```

## 2016 融 360-用户贷款风险预测

本次大赛要求参赛者基于由融 360 与平台上的金融机构合作的提供近 7 万贷款用户的基本身份信息、消费行为、银行还款等数据信息，建立准确的风险控制模型，来预测用户是否会逾期还款。

```
* Rank7：<https://github.com/hczheng/Rong360>
```

## 2016 CCF-020 优惠券使用预测

本次大赛要求参赛者基于给定的用户在 2016 年 1 月 1 日至 2016 年 6 月 30 日之间真实线上线下消费行为，预测用户在 2016 年 7 月领取优惠券后 15 天以内是否核销。比赛评测指标采用 AUC，先对每个优惠券单独计算核销预测的 AUC 值，再对所有优惠券的 AUC 值求平均作为最终的评价标准。

```
* Rank1: <https://github.com/wepe/O2O-Coupon-Usage-Forecast>
```

## 2016 CCF-农产品价格预测

本次大赛要求参赛者基于 2016 年 6 月以前的农产品价格数据，预测 7 月的农产品价格。本题目初赛基于全国各农场品交易市场的价格数据，复赛则加上天气等多源数据。

```
* Rank2: <https://github.com/xing89qs/CCF_Product> 
```

## 2016 CCF-客户用电异常

国家电网通过对用户及所属变压器进行异常监测，并通过现场检修人员根据异常情况对用户进行抽检，并反馈检查结果，如发现为窃电用户，将反馈窃电用户信息。本赛题要求参赛者通过提供的相关数据与检查人员检查结果，建立窃电检测模型，识别用户窃电行为。

```
* Rank4: <https://github.com/AbnerYang/2016CCF-StateGrid>
```

## 2016 CCF-搜狗的用户画像比赛

本题目初赛时要求参赛者基于给出的 2 万用户的百万级搜索词，以及经过调查得到的真实性别、年龄段、学历这一训练集，通过机器学习、数据挖掘技术构建分类算法对另外 2 万人群的搜索关键词进行分析，并给出其性别、年龄段、学历等用户属性信息。复赛时，训练集与测试集规模均扩展至 10 万用户。

```
* Rank1: <https://github.com/hengchao0248/ccf2016_sougou> 

* Rank3: <https://github.com/AbnerYang/2016CCF-SouGou> 

* Rank5: <https://github.com/dhdsjy/2016_CCFsougou> 
```

## 2016 CCF-联通的用户轨迹

精准营销是互联网营销和广告营销的新方向，特别是在用户身处特定的地点、商户，如何根据用户画像进行商户和用户的匹配，并将相应的优惠和广告信息通过不同渠道进行推送，成为了很多互联网和非互联网企业的新发展方向。本赛题以其中一个营销场景为例，要求参赛者基于提供的用户位置信息、商户分类与位置信息等数据，完成用户画像的刻画并进行商户匹配。

```
* RankX: <https://github.com/xuguanggen/2016CCF-unicom>
```

## 2016 CCF-Human or Robots

仅 2016 上半年，AdMaster 反作弊解决方案认定平均每天能有高达 28% 的虚假流量，即由机器人模拟和黑 IP 等手段导致的非人恶意流量。本赛题要求参赛者通过用户行为日志，自动检测出这些虚假流量。

```
* Rank6: <https://github.com/pickou/ccf_human_or_robot>
```

## 菜鸟-需求预测与分仓规划

本赛题要求参赛者以历史一年海量买家和卖家的数据为依据，预测某商品在未来二周全国和区域性需求量。参赛者需要用数据挖掘技术和方法精准刻画商品需求的变动规律，对未来的全国和区域性需求量进行预测，同时考虑到未来的不确定性对物流成本的影响，做到全局的最优化。比赛提供商品从 2014年 10 月 10 日到 2015 年 12 月 27 日的全国和区域分仓数据。

```
* Rank6: <https://github.com/wepe/CaiNiao-DemandForecast-StoragePlaning> 

* Rank10: <https://github.com/xing89qs/TianChi_CaiNiao_Season2>
```

# 自然语言处理（NLP）

## 2018 DC 达观-文本智能处理挑战 

此次比赛要求参赛者基于达观数据提供的一批长文本数据和分类信息，结合当下最先进的 NLP 和人工智能技术，深入分析文本内在结构和语义信息，构建文本分类模型，实现精准分类。比赛提供的数据包含训练数据集和测试数据集 2 个 csv 文件。

```
* Rank1: <https://github.com/ShawnyXiao/2018-DC-DataGrand-TextIntelProcess> 

* Rank4: <https://github.com/hecongqing/2018-daguan-competition> 

* Rank10: <https://github.com/moneyDboat/data_grand> 

* Rank18: <https://github.com/nlpjoe/daguan-classify-2018> 
```

## 智能客服问题相似度算法设计——第三届魔镜杯大赛

本次大赛要求参赛者基于拍拍贷提供的智能客服聊天机器人真实数据，以自然语言处理和文本挖掘技术为主要探索对象，利用这些资源开发一种提高智能客服的识别能力和服务质量的算法。

```
* rank6：<https://github.com/qrfaction/paipaidai>

* rank12：https://www.jianshu.com/p/827dd447daf9  
&emsp;&emsp;<https://github.com/LittletreeZou/Question-Pairs-Matching>

* Rank16：<https://github.com/guoday/PaiPaiDai2018_rank16> 
```

## 2018JD Dialog Challenge 任务导向型对话系统挑战赛

本次大赛要求参赛者基于京东用户与京东人工客服真实对话数据（脱敏后）以及给定的对话数据进行分析，构建端到端的任务驱动型多轮对话系统，输出满足用户需求的答案——该答案需要能正确、完整且高效地解决问题，为用户带来简单、省心、智能的购物咨询体验。

```
* Rank3: <https://github.com/zengbin93/jddc_solution_4th>
```

## 2018CIKM AnalytiCup – 阿里小蜜机器人跨语言短文本匹配算法竞赛

本次大赛关注短文本匹配在语言适应的问题，源语言为英语，目标语言为西班牙语。比赛要求参赛者建立跨语言短文本匹配模型，来提升智能客服机器人的能力。

```
* Rank2: <https://github.com/zake7749/Closer> 

* Rank12：<https://github.com/Leputa/CIKM-AnalytiCup-2018> 

* Rank18: <https://github.com/VincentChen525/Tianchi/tree/master/CIKM%20AnalytiCup%202018>
```

另外，Smilexuhc 还为大家提供了两篇经验文章，大家感兴趣的话可以一并收藏向前辈们取取经。

## 经验文章

* [《介绍 featexp一个帮助理解特征的工具包》](http://www.sohu.com/a/273552971_129720)

* 《Ask Me Anything session with a Kaggle Grandmaster Vladimir I. Iglovikov》PDF：<https://pan.baidu.com/s/1XkFwko_YrI5TfjjIai7ONQ>

Via：<https://github.com/Smilexuhc/Data-Competition-TopSolution>

# Reference

* [**2018 年度 GtiHub 开源项目 TOP 25：数据科学 & 机器学习**](https://www.leiphone.com/news/201901/G09wwNYYejYbEvgM.html)

* [**Google、亚马逊、微软 、阿里巴巴开源软件一览**](https://mp.weixin.qq.com/s/SGcrUAeGJRShgL9xT7m5QA?spm=ata.13261165.0.0.e65943b9K4tpac)

* [**Facebook、微信团队、Twitter、微软开源软件列表一览**](https://mp.weixin.qq.com/s/HzuGFCIgwl-MIWfabQAAuQ?spm=ata.13261165.0.0.e65943b9K4tpac)

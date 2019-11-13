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

# RNN机器翻译简史

在Transformer模型被提出之前，机器翻译一直是以RNN为主。

使用的工具是著名的RNN的两个改进版，1997年提出的长短时记忆网络LSTM和2014年提出的门控循环单元GRU。这三种实现均在torch.nn包中有提供。

应用这两项工具，2014年成为机器翻译的突破性一年。
2014年，主流的机器翻译方法seq2seq被提出，论文为《Sequence to sequence learning with neural networks》。同年，机器翻译中使用编码器-解码器的方案在论文《Learning phrase representations using rnn encoder-decoder for statistical machine translation》中被讨论。

同样是差不多的一批人，Bahdanau等人提出了主流的NMT方法，论文为《Neural machine translation by jointly learning to align and translate》将Attention机制引入到机器翻译中来。

NMT翻译方法成熟之后，Google团队迅速将其工程化，论文《Google’s neural machine translation system: Bridging the gap between human and machine translation》中介绍了他们的实现方法。后面有研究提高效率的《Effective approaches to attention based neural machine translation》，和解决限制的《Exploring the limits of language modeling》。

说了这么多，我们主要记住三个词就好，分别是编码器，解码器，注意力机制。这一讲的主题Transformer模型仍然没有超出这三点。

后面对于Attention的机制越来越热门，比如《Structured attention networks》。但是这些Attention也基本上是跟RNN结合在一起的。

Attention中值得一提的是自注意力机制self-attention，这是在机器阅读理解等领域被广泛证明的技术。

Transformer的基本结构如下：

![avatar](https://upload-images.jianshu.io/upload_images/6963844-676d65943cba6593.png?imageMogr2/auto-orient/strip|imageView2/2/w/384/format/webp)

在Transformer中主要使用的两种Attention机制如下：

![avatar](http://mmbiz.qpic.cn/mmbiz_png/KmXPKA19gWibp1593y9ib5hyUv34YYrkDnOaEYTg1FcozIHx6MFtHTNRHlQLEAYTf9UTudqRvepQTktXq5YkLVXA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

# Transformer先跑起来玩一玩

下面我们以PyTorch官方教程的Transformer例子为例，让大家先搭建起来一个可以跑可以玩的模型。

## Positional Encoding

Transformer对于位置信息的编码是通过正弦曲线来完成的。我们来看代码：
```python
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
```
## Transformer层的引入

现在在torch.nn中，TransformerEncoder和TransformerDecoder也像LSTM和GRU一样提供了。我们可以将其组织一下，和上面的位置编码器组合在一起：
```python
class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output
```
我们通过调用方法看下几个参数的含义：
```python
ntokens = len(TEXT.vocab.stoi) # the size of vocabulary
emsize = 200 # embedding dimension
nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
```

## 词嵌入层

ntokens是词表的大小，emsize是词嵌入层的维数，对应于模型中的形参ninp：
```python
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)
```
从上面可以看到，encoder是词嵌入层，词表大小是ntoken,也就是len(TEXT.vocab.stoi)，而词嵌入层的维度是200维。

torch.nn.Embedding的完整定义如下：
```python
CLASS torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None)
```
而decoder是个全连接层，输入是200，输出是len(TEXT.vocab.stoi)。
```python
TransformerEncoder
PyTorch中实现Transformer

torch.nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu')
```
其中的参数含义：

* d_model – 必选参数，输入的特征数  
* nhead – 必选参数，multi head attention models有多少个head.  
* dim_feedforward – 反馈网络的维数，默认是2048维  
* dropout – dropout值，缺省为0.1  
* activation – 中间层的激活函数，可选值是relu和gelu

对照本例，我们看看这些参数给的是什么：
```python
encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
```
ninp又是刚才的emsize，200维的词嵌入式维数。

nhead本例中是2，nhid层仍然选200维。dropout和激活函数保持默认值不变。

一个Encoder一般不够用，我们通常会多堆几层。TransformerEncoder是由多个TransformerEncoderLayer所构成：
```python
self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
```
本例中我们nlayers取2层。

## mask
建模做好之后，数据输入之时，TransformerEncoder还需要一个mask：
```python
        output = self.transformer_encoder(src, self.src_mask)
```
这个mask是用来遮挡住部分视野的，我们来看看代码中实现的mask是个什么鬼？

如果没有指定的话，就给它生成一个：
```python
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
```
生成的过程如下：
```python
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
```

首先是通过triu函数生成一个上对角阵，然后将其转置，生成的结果是这样的：
```python
 tensor([[ True, False, False,  ..., False, False, False],
        [ True,  True, False,  ..., False, False, False],
        [ True,  True,  True,  ..., False, False, False],
        ...,
        [ True,  True,  True,  ...,  True, False, False],
        [ True,  True,  True,  ...,  True,  True, False],
        [ True,  True,  True,  ...,  True,  True,  True]])
```
然后给这个矩阵赋值成负无穷和0，变成这样：
```python
mask2= tensor([[0., -inf, -inf, ..., -inf, -inf, -inf],
[0., 0., -inf, ..., -inf, -inf, -inf],
[0., 0., 0., ..., -inf, -inf, -inf],
...,
[0., 0., 0., ..., 0., -inf, -inf],
[0., 0., 0., ..., 0., 0., -inf],
[0., 0., 0., ..., 0., 0., 0.]])
```
## 准备数据

我们取Wikitext2为训练数据：
```python
import torchtext
from torchtext.data.utils import get_tokenizer
TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
                            init_token='<sos>',
                            eos_token='<eos>',
                            lower=True)
train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
TEXT.build_vocab(train_txt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
这个数据是用作语言模型用途的，也就是训练在给定前面的token序列下，推理后面该出现的最可能的token的过程。

然后将其拆分成若干个批次：
```python
def batchify(data, bsz):
    # print(data.examples[0].text)
    data = TEXT.numericalize([data.examples[0].text])
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    print(nbatch)
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)
```
将其分为训练集、测试集和验证集：
```python
batch_size = 20
eval_batch_size = 10
train_data = batchify(train_txt, batch_size)
val_data = batchify(val_txt, eval_batch_size)
test_data = batchify(test_txt, eval_batch_size)
```
接着我们还要为Transformer模型做准备，将源数据切成bptt长度的小段：
```python
bptt = 35
def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target
```
## 训练过程

建模齐备之后，后面PyTorch的训练过程就是例行公事地获取数据，model，criterion，backward，step几步曲了。
```python
import time
def train():
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    ntokens = len(TEXT.vocab.stoi)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
```

## 验证过程

同样，验证过程也是大同小异：
```python
def evaluate(eval_model, data_source):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    ntokens = len(TEXT.vocab.stoi)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            output = eval_model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)
```
## 多次训练改进效果

最后，我们可以多训练几次，看看效果的提升：
```python
best_val_loss = float("inf")
epochs = 3 # The number of epochs
best_model = None

for epoch in range(1, epochs + 1):
    train()
    val_loss = evaluate(model, val_data)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

    scheduler.step()
```
这样，一个Transformer训练语言模型的完整过程就算是圆满完成了。

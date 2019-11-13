---
title: PyTorch快餐教程(2)-MNIST问题处理
date: 2019-01-05
categories: Tensorflow
tags:
- Tensorflow
---

# 背景简介

## 使用深度学习解决问题的通用方法

在解决各种实际问题之前，我们先要提纲契领地了解使用深度学习解决问题的方法。

<!-- more -->

首先，我们需要强调一点，深度学习学习的是人类可以标记的知识。如果人解决都困难的话，需要通过别的方法首先让人先能够解决。

举例说明。比如要做语音识别，结果杂音特别多，声音模糊到人类都听不清内容，那么指望通过深度学习来识别出来基本上没戏。但是，深度学习并不是唯一的手段，我们还有其他的模型和工具可以用。比如声学的工具，或者是浅层的机器学习等，将这段音频变成人类听起来比较舒服了，再用深度学习的方法解决就相对容易了。

再举个例子，比如给出一张女生的照片，让认识她的人识别出是不是她，这个可以做到。但是，要是问这个女生当时在想什么，这事儿就没谱了。什么人也没有办法仅凭照片就得出这样的信息。这就属于深度学习也解决不了的问题，因为根本没有可靠有效的数据可以学习。

理解了上面这些之后，我们就可以容易地理解下面的三步法的原理了：

* 首先找到人类能够准确解决这个问题的方法。  
* 为每份数据进行人工标注。  
* 选用深度学习工具对标注好的数据进行学习。 

大家可能还是觉得不够直观，下面举例说明。

## MNIST手写数字识别问题

数据科学研究院NIST收集了大量的手写的数字，并且对其进行了剪裁、位置调整等预处理的工作。深度学习三巨头之一的Yann LeCun从中精选了一个子集，这就是深度学习著名的入门教材MNIST数据集。这个数据可以从Yann LeCun的网站上免费下载：<http://yann.lecun.com/exdb/mnist/>。

我们应用这节所介绍的三步法来看看这个问题该如何解决。

* 第一步，NIST和Yann LeCun其实已经帮我们做了预处理了，每个数字独立存放在一张28像素乘以28像素的图片中。我们人类看了这些图片，能够认出来每个图片表示什么数字。说明这个问题是深度学习可以解决的。  
* 第二步，既然人工可以认出来这些图片表示的数字，下一步就进入人工标注阶段，我们给每张图片注明所对应的数字。  
* 第三步，有了标注的数据之后，我们就可以选择一个深度学习的模型来处理它了。

MNIST网站上一共提供了4个文件，两个训练文件，两个测试文件。训练文件有两个，一个是由28乘28图片组成的图片的压缩包：<http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz>，另一个是对图片人工标注好的图片对应的数字值：<http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz>。

训练集合一共有60000张图片，相应的有60000个人工标注好的数据。测试集合的格式与训练集一模一样，不过只有10000张图片和标注。

首先我们看下简单的，将train-labels-idx1-ubyte.gz用解压缩工具如gunzip命令解压，获得60008字节的train-labels-idx1-ubyte文件。除了8字节的文件头，刚好是60000个标注好的数字。

我们写段代码来读取这60000个标记的数据吧，代码如下：
```python
f = open('./train-labels-idx1-ubyte', 'rb')
# 略过8字节的文件头
f.seek(8)
# 文件头之后，每个字节代表一个0-9之间的数字
data = f.read(1)
print(data)
f.close()
```

seek(8)是跳过文件头。读取之后，每个字节都代表一个标记好的数据。我们上边的代码只读第一个数，值为5。

train-images-idx3-ubyte.gz解压之后，得到的train-images-idx3-ubyte文件的大小为47040016字节。28*28*60000 = 47040000。刚好是28*28的图片60000张，加上一个16个字节的文件头。

我们来写段代码，读出第一个图像，代码如下：
```python
import numpy as np
import matplotlib.pyplot as plt
file_train_image = open('./train-images-idx3-ubyte','rb')
# 图像文件的头是16个字节，略过不读
file_train_image.seek(16)
image1 = file_train_image.read(28*28)
image2 = np.zeros(28*28)
# 将值转换成灰度
for i in range(0,28*28-1):
    image2[i] = image1[i]/256
image2 = image2.reshape(28,28)
print(image2)
plt.imshow(image2, cmap='binary')
plt.show()
file_train_image.close()
```
首先我们通过seek(16)略过16字节的文件头。然后再读取28乘以28字节，就是这个图片的灰度值，0是全白色，255是全黑色。

然后我们借助numpy工具，将读取的二进制字节流转换成矩阵。绘图时需要的是灰度值，是(0,1)之间的一个小数，所以我们将读取的值除以256换算成灰度。运行结果如图2.4所示，对应到之前我们读取的标注数据，这是个5。

为了方便起见，我们使用了一个现成的例子，在实际工作中，收集和标注数据通常也不是我们的工作。但是，大家完全可以自己找数据。只要能够按照原始数据加上人工标注数据一一对应的方式，就成功地完成了第二步。

下面我们把刚才读取数据的代码稍整理一下，图像数据读取到x_train列表中，对应的标签读取到y_train中，代码如下：
```python
import numpy as np
import matplotlib.pyplot as plt

f = open('./train-labels-idx1-ubyte', 'rb')
f.seek(8)
data = f.read(60000)
y_train = np.zeros(60000)
for i in range(60000):
    y_train[i] = data[i]
f.close()

file_train_image = open('./train-images-idx3-ubyte', 'rb')
file_train_image.seek(16)

X_train = []

for i in range(60000):
    image1 = file_train_image.read(28 * 28)
    image2 = np.zeros(28 * 28)
    for i2 in range(0, 28 * 28 - 1):
        image2[i2] = image1[i2] / 255
    image2 = image2.reshape(28, 28)
    X_train.append(image2)

file_train_image.close()
```

数据我们已经充分理解了，下面我们开始用深度学习来处理这个问题。

# PyTorch的建模过程

做为对照，我们先看下Keras中的建模部分：
```python
model = Sequential()

model.add(Dense(units=121, input_dim=28*28))
model.add(Activation('relu'))
model.add(Dense(units=10))
model.add(Activation('softmax'))
```
对于不熟悉Keras的同学，我来解释下上面的含义:输入是28*28=784个元素，然后有一个121节点的隐藏层。接着增加一个relu激活函数，最后输出10个元素，用softmax对应到10个分类中。

PyTorch中采用继承torch.nn.Module，实现其中的__init__和forward两个函数的方式来实现建模的功能，我们对照翻译下上面的语句：
```python
# PyTorch采用结构化写法
class TestNet(t.nn.Module):
    # 初始化传入输入、隐藏层、输出三个参数
    def __init__(self, in_dim, hidden, out_dim):
        super(TestNet, self).__init__()
        self.layer1 = t.nn.Sequential(t.nn.Linear(in_dim, hidden), t.nn.ReLU(True))
        self.layer2 = t.nn.Linear(hidden, out_dim)

    # 传入计算值的函数，真正的计算在这里
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

# 输入28*28，隐藏层121，输出10类
model = TestNet(28 * 28, 121, 10)
```

# 训练和验证

## 编译

我们还是和Keras的编译语句对照一下：
```
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
```
解释一下：

第一个loss是损失函数，指定的是分类的交叉熵。我们翻译成PyTorch：
```
loss = t.nn.CrossEntropyLoss()(out, y)
```
第二个optimizer选取了随机梯度下降，我们也翻译成PyTorch:
```
optimizer = t.optim.SGD(model.parameters(), lr=learning_rate)
```
第三个参数是打印出准确率信息，这个PyTorch没有简单的对译，得需要编程来实现。

## 训练

训练更能体会到Keras的简洁了，Keras的训练只需要一句话：
```python
model.fit(X_train, y_train,
          batch_size=64,
          epochs=20,
          verbose=1,
          validation_data=(X_test, y_test))
```
用PyTorch的话我们需要多写几行代码，但是其实逻辑也很简单，我们看下PyTorch的训练过程：
```python
for epoch in range(num_epochs):
    X = t.autograd.Variable(t.from_numpy(X_train))
    y = t.autograd.Variable(t.from_numpy(y_train))

    # 正向传播

    # 用神经网络计算10类输出结果
    out = model(X)
    # 计算神经网络结果与实际标签结果的差值
    loss = t.nn.CrossEntropyLoss()(out, y)

    # 反向梯度下降

    # 清空梯度
    optimizer.zero_grad()
    # 根据误差函数求导
    loss.backward()
    # 进行一轮梯度下降计算
    optimizer.step()
```
其实核心计算也就是后面的两句：
```python
    loss.backward()    
    optimizer.step()
```
刚才那一版，我们是把60000条数据都一次性拿来训练了。等于说是我们还没有实现keras中的batch_size=64这样分批次的功能。这也很easy，我们手动分下批次就好了：
```python
X_train_size = len(X_train)

for epoch in range(num_epochs):

    X = t.autograd.Variable(t.from_numpy(X_train))
    y = t.autograd.Variable(t.from_numpy(y_train))

    i = 0
    while i < X_train_size:
        #取一个新批次的数据
        X0 = X[i:i+batch_size]
        y0 = y[i:i+batch_size]
        i += batch_size

        # 正向传播

        ## 用神经网络计算10类输出结果
        out = model(X0)
        ## 计算神经网络结果与实际标签结果的差值
        loss = t.nn.CrossEntropyLoss()(out, y0)

        # 反向梯度下降

        ## 清空梯度
        optimizer.zero_grad()
        ## 根据误差函数求导
        loss.backward()
        ## 进行一轮梯度下降计算
        optimizer.step()
```

## 验证

在验证时，keras只用了一句话：
```
score = model.evaluate(X_test, y_test, verbose=1)
```
对于PyTorch的话，虽然多了几句，还是也还是很好理解的：
```python
# 验证部分

## 将模型设为验证模式
model.eval()

X_val = t.autograd.Variable(t.from_numpy(X_test))
y_val = t.autograd.Variable(t.from_numpy(y_test))

## 用训练好的模型计算结果
out_val = model(X_val)
loss_val = t.nn.CrossEntropyLoss()(out_val, y_val)

# 求出最大的元素的位置
_, pred = t.max(out_val,1)
# 将预测值与标注值进行对比
num_correct = (pred == y_val).sum()
```

## 完整代码

下面我们看一下完整的代码：
```python
import numpy as np
import torch as t


# 读取图片对应的数字
def read_labels(filename, items):
    file_labels = open(filename, 'rb')
    file_labels.seek(8)
    data = file_labels.read(items)
    y = np.zeros(items, dtype=np.int64)
    for i in range(items):
        y[i] = data[i]
    file_labels.close()
    return y


y_train = read_labels('./train-labels-idx1-ubyte', 60000)
y_test = read_labels('.t10k-labels-idx1-ubyte', 10000)


# 读取图像
def read_images(filename, items):
    file_image = open(filename, 'rb')
    file_image.seek(16)

    data = file_image.read(items * 28 * 28)

    X = np.zeros(items * 28 * 28, dtype=np.float32)
    for i in range(items * 28 * 28):
        X[i] = data[i] / 255
    file_image.close()
    return X.reshape(-1, 28 * 28)


X_train = read_images('./train-images-idx3-ubyte', 60000)
X_test = read_images('./t10k-images-idx3-ubyte', 10000)

# 超参数

# 训练轮数
num_epochs = 1000
# 学习率
learning_rate = 1e-3
# 批量大小
batch_size = 64

# 测试网络
class TestNet(t.nn.Module):
    # 初始化传入输入、隐藏层、输出三个参数
    def __init__(self, in_dim, hidden, out_dim):
        super(TestNet, self).__init__()
        self.layer1 = t.nn.Sequential(t.nn.Linear(in_dim, hidden), t.nn.ReLU(True))
        self.layer2 = t.nn.Linear(hidden, out_dim)

    # 传入计算值的函数，真正的计算在这里
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

# 输入28*28，隐藏层121，输出10类
model = TestNet(28 * 28, 121, 10)

#model = t.nn.Linear(28 * 28, 10)

# 优化器仍然选随机梯度下降
optimizer = t.optim.SGD(model.parameters(), lr=learning_rate)

X_train_size = len(X_train)

for epoch in range(num_epochs):
    # 打印轮次：
    print('Epoch:',epoch)

    X = t.autograd.Variable(t.from_numpy(X_train))
    y = t.autograd.Variable(t.from_numpy(y_train))

    i = 0
    while i < X_train_size:
        #取一个新批次的数据
        X0 = X[i:i+batch_size]
        y0 = y[i:i+batch_size]
        i += batch_size

        # 正向传播

        ## 用神经网络计算10类输出结果
        out = model(X0)
        ## 计算神经网络结果与实际标签结果的差值
        loss = t.nn.CrossEntropyLoss()(out, y0)

        # 反向梯度下降

        ## 清空梯度
        optimizer.zero_grad()
        ## 根据误差函数求导
        loss.backward()
        ## 进行一轮梯度下降计算
        optimizer.step()

    print(loss.item())

# 验证部分

## 将模型设为验证模式
model.eval()

X_val = t.autograd.Variable(t.from_numpy(X_test))
y_val = t.autograd.Variable(t.from_numpy(y_test))

## 用训练好的模型计算结果
out_val = model(X_val)
loss_val = t.nn.CrossEntropyLoss()(out_val, y_val)

print(loss_val.item())

# 求出最大的元素的位置
_, pred = t.max(out_val,1)
# 将预测值与标注值进行对比
num_correct = (pred == y_val).sum()

print(num_correct.data.numpy()/len(y_test))
```
实测效果：在训练100轮时，准确率在93%~94%左右，训练1000轮以后，准确率在97%左右。

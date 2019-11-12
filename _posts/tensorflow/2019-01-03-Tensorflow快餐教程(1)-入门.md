---
title: Tensorflow快餐教程(1)-入门
date: 2019-01-03
categories: Tensorflow
tags:
- DNN
- Tensorflow
---

# 背景

根据几本tensorflow的书，发现有些样例代码所用的API已经过时了。快餐教程系列希望能够尽可能降低门槛，少讲，讲透，首先参考网上的一些教程，直接展示用tensorflow实现MNIST手写识别的例子。然后基础知识再慢慢讲述。

<!-- more -->

# Tensorflow安装速成教程

由于Python是跨平台的语言，所以在各系统上安装tensorflow都是一件相对比较容易的事情。GPU加速的事情我们后面再说。

## Linux平台安装tensorflow

我们以Ubuntu 16.04版为例，首先安装python3和pip3。pip是python的包管理工具。
```
sudo apt install python3
sudo apt install python3-pip
```
然后就可以通过pip3来安装tensorflow:
```
pip3 install tensorflow --upgrade
```

## MacOS安装tensorflow

建议使用Homebrew来安装python。
```
brew install python3
```
安装python3之后，还是通过pip3来安装tensorflow.
```
pip3 install tensorflow --upgrade
```

## Windows平台安装Tensorflow

Windows平台上建议通过Anaconda来安装tensorflow，下载地址在：<https://www.anaconda.com/download/#windows>

然后打开Anaconda Prompt，输入：
```
conda create -n tensorflow pip
activate tensorflow
pip install --ignore-installed --upgrade tensorflow
```
这样就安装好了Tensorflow。

我们迅速来个例子试下好不好用：
```python
import tensorflow as tf
a = tf.constant(1)
b = tf.constant(2)

c = a * b

sess = tf.Session()

print(sess.run(c))
```

输出结果为2.

Tensorflow顾名思义，是一些Tensor张量的流组成的运算。运算需要一个Session来运行。如果print(c)的话，会得到
```
Tensor("mul_1:0", shape=(), dtype=int32)
```
就是说这是一个乘法运算的Tensor，需要通过Session.run()来执行。

# 入门捷径：线性回归

我们首先看一个最简单的机器学习模型，线性回归的例子。

线性回归的模型就是一个矩阵乘法：
```
tf.multiply(X, w)
```
然后我们通过调用Tensorflow计算梯度下降的函数tf.train.GradientDescentOptimizer来实现优化。

我们看下这个例子代码，只有30多行，逻辑还是很清晰的。例子来自github上大牛的工作：<https://github.com/nlintz/TensorFlow-Tutorials>。

```python
import tensorflow as tf
import numpy as np

trX = np.linspace(-1, 1, 101)
trY = 2 * trX + np.random.randn(*trX.shape) * 0.33 # 创建一些线性值附近的随机值

X = tf.placeholder("float") 
Y = tf.placeholder("float")

def model(X, w):
    return tf.multiply(X, w) # X*w线性求值，非常简单

w = tf.Variable(0.0, name="weights") 
y_model = model(X, w)

cost = tf.square(Y - y_model) # 用平方误差做为优化目标

train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost) # 梯度下降优化

# 开始创建Session干活！
with tf.Session() as sess:
    # 首先需要初始化全局变量，这是Tensorflow的要求
    tf.global_variables_initializer().run()

    for i in range(100):
        for (x, y) in zip(trX, trY):
            sess.run(train_op, feed_dict={X: x, Y: y})

    print(sess.run(w)) 
```
最终会得到一个接近2的值，比如我这次运行的值为1.9183811

# 多种方式搞定手写识别

线性回归不过瘾，我们直接一步到位，开始进行手写识别。

![avatar](https://ask.qcloudimg.com/http-save/yehe-1205621/zqk26vr3zx.jpeg?imageView2/2/w/1620)

我们采用深度学习三巨头之一的Yann Lecun教授的MNIST数据为例。如上图所示，MNIST的数据是28x28的图像，并且标记了它的值应该是什么。

## 线性模型：logistic回归

算上注释和空行，一共加起来30行左右，我们就可以解决手写识别这么困难的问题啦！请看代码：

```python
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w):
    return tf.matmul(X, w) # 模型还是矩阵乘法

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

X = tf.placeholder("float", [None, 784]) # create symbolic variables
Y = tf.placeholder("float", [None, 10])

w = init_weights([784, 10]) 
py_x = model(X, w)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)) # 计算误差
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct optimizer
predict_op = tf.argmax(py_x, 1) 

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i in range(100):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
        print(i, np.mean(np.argmax(teY, axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX})))
```
经过100轮的训练，我们的准确率是92.36%。

## 无脑的浅层神经网络

用了最简单的线性模型，我们换成经典的神经网络来实现这个功能。神经网络的图如下图所示。

![avatar](http://www.kongzhi.net/uploads/images/cases/2007/9/1190777620.gif)

我们还是不管三七二十一，建立一个隐藏层，用最传统的sigmoid函数做激活函数。其核心逻辑还是矩阵乘法，这里面没有任何技巧。
```
    h = tf.nn.sigmoid(tf.matmul(X, w_h)) 
    return tf.matmul(h, w_o) 
```
完整代码如下，仍然是40多行，不长：
```python
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# 所有连接随机生成权值
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w_h, w_o):
    h = tf.nn.sigmoid(tf.matmul(X, w_h)) 
    return tf.matmul(h, w_o) 

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

w_h = init_weights([784, 625])
w_o = init_weights([625, 10])

py_x = model(X, w_h, w_o)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)) # 计算误差损失
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct an optimizer
predict_op = tf.argmax(py_x, 1)

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i in range(100):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
        print(i, np.mean(np.argmax(teY, axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX})))
```
第一轮运行，我这次的准确率只有69.11% ，第二次就提升到了82.29%。最终结果是95.41%，比Logistic回归的强！

请注意我们模型的核心那两行代码，完全就是无脑地全连接做了一个隐藏层而己，这其中没有任何的技术。完全是靠神经网络的模型能力。

## 深度学习时代的方案 - ReLU和Dropout显神通

上一个技术含量有点低，现在是深度学习的时代了，我们有很多进步。比如我们知道要将sigmoid函数换成ReLU函数。我们还知道要做Dropout了。于是我们还是一个隐藏层，写个更现代一点的模型吧：
```python
    X = tf.nn.dropout(X, p_keep_input)
    h = tf.nn.relu(tf.matmul(X, w_h))

    h = tf.nn.dropout(h, p_keep_hidden)
    h2 = tf.nn.relu(tf.matmul(h, w_h2))

    h2 = tf.nn.dropout(h2, p_keep_hidden)

    return tf.matmul(h2, w_o)
```
除了ReLU和dropout这两个技巧，我们仍然只有一个隐藏层，表达能力没有太大的增强。并不能算是深度学习。

```python
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden): 
    X = tf.nn.dropout(X, p_keep_input)
    h = tf.nn.relu(tf.matmul(X, w_h))

    h = tf.nn.dropout(h, p_keep_hidden)
    h2 = tf.nn.relu(tf.matmul(h, w_h2))

    h2 = tf.nn.dropout(h2, p_keep_hidden)

    return tf.matmul(h2, w_o)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

w_h = init_weights([784, 625])
w_h2 = init_weights([625, 625])
w_o = init_weights([625, 10])

p_keep_input = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()

    for i in range(100):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          p_keep_input: 0.8, p_keep_hidden: 0.5})
        print(i, np.mean(np.argmax(teY, axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX,
                                                         p_keep_input: 1.0,
                                                         p_keep_hidden: 1.0})))
```
从结果看到，第二次就达到了96%以上的正确率。后来就一直在98.4%左右游荡。仅仅是ReLU和Dropout，就把准确率从95%提升到了98%以上。

# 卷积神经网络

真正的深度学习利器CNN，卷积神经网络出场。这次的模型比起前面几个无脑型的，的确是复杂一些。涉及到卷积层和池化层。这个是需要我们后面详细讲一讲了。

```python
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

batch_size = 128
test_size = 256

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
    l1a = tf.nn.relu(tf.nn.conv2d(X, w,                       # l1a shape=(?, 28, 28, 32)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],              # l1 shape=(?, 14, 14, 32)
                        strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2,                     # l2a shape=(?, 14, 14, 64)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],              # l2 shape=(?, 7, 7, 64)
                        strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)

    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3,                     # l3a shape=(?, 7, 7, 128)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],              # l3 shape=(?, 4, 4, 128)
                        strides=[1, 2, 2, 1], padding='SAME')
    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])    # reshape to (?, 2048)
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(l3, w4))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_o)
    return pyx

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX = trX.reshape(-1, 28, 28, 1)  # 28x28x1 input img
teX = teX.reshape(-1, 28, 28, 1)  # 28x28x1 input img

X = tf.placeholder("float", [None, 28, 28, 1])
Y = tf.placeholder("float", [None, 10])

w = init_weights([3, 3, 1, 32])       # 3x3x1 conv, 32 outputs
w2 = init_weights([3, 3, 32, 64])     # 3x3x32 conv, 64 outputs
w3 = init_weights([3, 3, 64, 128])    # 3x3x32 conv, 128 outputs
w4 = init_weights([128 * 4 * 4, 625]) # FC 128 * 4 * 4 inputs, 625 outputs
w_o = init_weights([625, 10])         # FC 625 inputs, 10 outputs (labels)

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()

    for i in range(100):
        training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX)+1, batch_size))
        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          p_keep_conv: 0.8, p_keep_hidden: 0.5})

        test_indices = np.arange(len(teX)) # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]

        print(i, np.mean(np.argmax(teY[test_indices], axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX[test_indices],
                                                         p_keep_conv: 1.0,
                                                         p_keep_hidden: 1.0})))
```

我们看下这次的运行数据：
```
0 0.95703125
1 0.9921875
2 0.9921875
3 0.98046875
4 0.97265625
5 0.98828125
6 0.99609375
```
在第6轮的时候，就跑出了99.6%的高分值，比ReLU和Dropout的一个隐藏层的神经网络的98.4%大大提高。因为难度是越到后面越困难。

在第16轮的时候，竟然跑出了100%的正确率：
```
7 0.99609375
8 0.99609375
9 0.98828125
10 0.98828125
11 0.9921875
12 0.98046875
13 0.99609375
14 0.9921875
15 0.99609375
16 1.0
```
综上，借助Tensorflow和机器学习工具，我们只有几十行代码，就解决了手写识别这样级别的问题，而且准确度可以达到如此程度。

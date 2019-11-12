---
title: Tensorflow快餐教程(10)-循环神经网络
date: 2019-01-03
categories: Tensorflow
tags:
- Tensorflow
---

# 循环神经网络

上节介绍了在图像和语音领域里大放异彩引发革命的CNN。但是，还有一类问题是CNN所不擅长的。这类问题的特点是上下文相关序列，比如理解文字。这时需要一种带有记忆的结构，于是，深度学习中的另一法宝RNN横空出世了。

<!-- more -->

大家还记得第8节中我们讲的人工神经网络的第二次复兴吗？没错，第二次复兴的标志正是1984年加州理工学院的物理学家霍普菲尔德实现了他于两年前提出的一种循环神经网络模型。这种网络被称为Hopfield网络。当时因为硬件条件的限制，Hopfield网络并没有得到广泛应用。而两年扣BP网络被重新发明，全连接前馈神经网络成为主流。RNN正是在Hopfield网络的基础上发展起来的。

![avatar](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-rolled.png)

RNN的图片都取自：https://colah.github.io/posts/2015-08-Understanding-LSTMs/

从图中我们可以看到，一个典型的循环神经网络神经元的结构，是在输入 $X_t$之外，A 与自己也有一个连接。

我们将其展开的话可能看得更清楚一些：

![avatar](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png)

也就是前一次输出的结果是对下一次输出有影响。

## LSTM

RNN中增加了对于之前状态的记忆项，不能直接使用之前BP网络的梯度下降的方法。但是基于该方法将循环项的输入都考虑进来，这个改进方法叫做BPTT算法（Back-Propagation Through Time）。

但是这种方法有个隐患，就是输入序列过长时会出现梯度消散问题（the vanishing gradient problem）。

于是一个改进算法LSTM(Long short-term memory)就增加了一个遗忘的机制。

LSTM的细节我们放到后面详细讲。我们先看看在Tensorflow中如何实现一个LSTM模型：
```python
def RNN(x, weights, biases):
    x = tf.unstack(x, timesteps, 1)

    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    return tf.matmul(outputs[-1], weights['out']) + biases['out']
```

第一步准备数据，第二步创建一个LSTMCell，第三步连成一个RNN网络，第四步矩阵乘输出。

下面我们还是以第1讲的例子来用LSTM来处理MNIST分类问题，第一时间有个可以运行的代码：
```python
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# 训练参数
learning_rate = 0.001
training_steps = 10000
batch_size = 128
display_step = 200

# 网络参数
num_input = 28 # MNIST data input (img shape: 28*28)
timesteps = 28 # timesteps
num_hidden = 128 # hidden layer num of features
num_classes = 10 # MNIST total classes (0-9 digits)

X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])

# 初始权值
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}

def RNN(x, weights, biases):
    x = tf.unstack(x, timesteps, 1)

    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    return tf.matmul(outputs[-1], weights['out']) + biases['out']

logits = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

# 定义损失和优化函数
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)

    for step in range(1, training_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))
```

## 门控循环单元GRU(Gated Recurrent Unit)

LSTM所使用的技术属于门控RNN（Gated RNN）技术。除了LSTM之外，还有一种应用广泛的门控RNN叫做GRU(Gated Recurrent Unit).

不同于1997年就发明的LSTM，GRU的技术比较新，提出在2014年。GRU与LSTM的不同在于，GRU同时可以控制『更新』门和『复位』门。

在Tensorflow中，使用tf.contrib.rnn.GRUCell来表示GRU单元。

到Tensorflow 1.8版本，一共支持5种单元，其中4种是LSTM单元，1种是GRU单元：

* tf.contrib.rnn.BasicRNNCell  
* tf.contrib.rnn.BasicLSTMCell  
* tf.contrib.rnn.GRUCell  
* tf.contrib.rnn.LSTMCell  
* tf.contrib.rnn.LayerNormBasicLSTMCell

## 双向循环神经网络

从前面的LSTM的结构我们可以看到，它是有方向的。GRU是在LSTM基础上的改良，也是如此。就像一个链表一样。

那么，我们如果想同时支持两个方向该怎么办？这就是双向循环神经网络。

我们还是先看核心代码：
```python
def BiRNN(x, weights, biases):
    x = tf.unstack(x, timesteps, 1)

    lstm_fw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    lstm_bw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    try:
        outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                              dtype=tf.float32)
    except Exception: 
        outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                        dtype=tf.float32)

    return tf.matmul(outputs[-1], weights['out']) + biases['out']
```
别的没什么变化，就是前向和后向各需要一个单元，然后调用static_bidirectional_rnn来运行网络。

最后是双向RNN训练MNIST的完整代码：
```python

from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

learning_rate = 0.001
training_steps = 10000
batch_size = 128
display_step = 200

num_input = 28 # MNIST data input (img shape: 28*28)
timesteps = 28 # timesteps
num_hidden = 128 # hidden layer num of features
num_classes = 10 # MNIST total classes (0-9 digits)

X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])

weights = {
     'out': tf.Variable(tf.random_normal([2*num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}

def BiRNN(x, weights, biases):

    x = tf.unstack(x, timesteps, 1)

    lstm_fw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    lstm_bw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    try:
        outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                              dtype=tf.float32)
    except Exception: # Old TensorFlow version only returns outputs not states
        outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                        dtype=tf.float32)

    return tf.matmul(outputs[-1], weights['out']) + biases['out']

logits = BiRNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)

    for step in range(1, training_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))
```

# 小结

到现在为止，CNN和RNN两大主角已经全部出场了。


---
title: Tensorflow-cnn快速上手
date: 2019-01-02
categories: Tensorflow
tags:
- DNN
- Tensorflow
- github
---

学习CNN 的方法, 就如同我们以前学习各种编程技术一样, 最快的学习路径, 就是找到一些合适的小例子, 迅速download下来, 边运行边调试, 先run起来.

# CNN 在 tensorboard 的展开图



# 编写自定义cnn函数, 编程就像写诗一样
```python
# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)
def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')
# 构建 cnn的每一层
# in_x 输入
# in_deep_num, 输入层的通道数, 一般是1(代表灰度图像), 或者3(代表rgb三个彩色通道), 
# in_deep_num, 也可以是上个卷积层的过滤核的总数
# kernel_size, 滑动卷积窗口的大小, 一般是5*5或者3*3
# kernel_num, 本卷积层里的过滤核的总数
def cnn(in_x, kernel_size, kernel_num):
    # 输入的维度
    in_shape = in_x.get_shape()
    # 输入图片的尺寸
    in_size_x = int( in_shape[1])
    in_size_y = int( in_shape[2])
    # 输入图片的深度
    in_deep_num = int( in_shape[3])
    # 拼接出: 输入层的名称
    layer_name = 'cnn_'+ str(in_size_x)+'x'+str(in_size_y)+'x'+str(in_deep_num)
    # 拼接出: 输出层的名称
    layer_name = layer_name+'-'+str(kernel_size)+'x'+str(kernel_size)+'x'+str(kernel_num)
    print layer_name
    with tf.name_scope(layer_name):
        # 初始化 权重 w和b
        weight = tf.Variable(tf.random_normal([kernel_size, kernel_size, in_deep_num,  kernel_num]), name='weight')
        bias = tf.Variable(tf.random_normal([kernel_num]), name='bias')
        # 卷积
        conv = conv2d(in_x, weight, bias)
        # 池化
        return maxpool2d(conv)
```

## 使用方法
```python
    #输入x, 单通道图像, 本层的卷积核5*5, 共有32个核, 输出
    conv1 = cnn( x, 5, 32 )
    #输入conv1, 本层的卷积核5*5, 共有64核,输出
    conv2 = cnn( conv1, 5, 64)
    # 拉伸为一维向量
    fc1 = reshape_vector(conv2)
```

# tensorboard使用方法
加快刷新速度为2秒
tensorboard --logdir=log1 --reload_interval=2

浏览器 刷新, http://127.0.0.1:6006/

# 可运行的源代码
```python
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/convolutional_network.ipynb

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)
```

```python
Extracting ./MNIST_data/train-images-idx3-ubyte.gz
Extracting ./MNIST_data/train-labels-idx1-ubyte.gz
Extracting ./MNIST_data/t10k-images-idx3-ubyte.gz
Extracting ./MNIST_data/t10k-labels-idx1-ubyte.gz
```

```python
# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

```

```python
# 清空以前的旧图
tf.reset_default_graph()

# tf Graph input
with tf.name_scope('input'):
    x = tf.placeholder("float", [None, n_input], name='x_input')
    y = tf.placeholder("float", [None, n_classes], name='y_input')
    keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# 构建 cnn的每一层
# in_x 输入
# in_deep_num, 输入层的通道数, 一般是1(代表灰度图像), 或者3(代表rgb三个彩色通道), 
# in_deep_num, 也可以是上个卷积层的过滤核的总数

# kernel_size, 滑动卷积窗口的大小, 一般是5*5或者3*3
# kernel_num, 本卷积层里的过滤核的总数
def cnn(in_x, kernel_size, kernel_num):
    # 输入的维度
    in_shape = in_x.get_shape()
    # 输入图片的尺寸
    in_size_x = int( in_shape[1])
    in_size_y = int( in_shape[2])
    # 输入图片的深度
    in_deep_num = int( in_shape[3])

    # 拼接出: 输入层的名称
    layer_name = 'cnn_'+ str(in_size_x)+'x'+str(in_size_y)+'x'+str(in_deep_num)
    # 拼接出: 输出层的名称
    layer_name = layer_name+'-'+str(kernel_size)+'x'+str(kernel_size)+'x'+str(kernel_num)

    print layer_name

    with tf.name_scope(layer_name):
        # 初始化 权重 w和b
        weight = tf.Variable(tf.random_normal([kernel_size, kernel_size, in_deep_num,  kernel_num]), name='weight')
        bias = tf.Variable(tf.random_normal([kernel_num]), name='bias')

        # 卷积
        conv = conv2d(in_x, weight, bias)

        # 池化
        return maxpool2d(conv)


# 把输入input_tensor, 拉伸为一维向量
def reshape_vector(input_tensor):
    dims = input_tensor.get_shape().as_list()
    print dims

    vector_length =1
    # 舍弃第1个维度
    for dim in dims[1:] :
        vector_length = vector_length *dim

    print vector_length
    vector = tf.reshape(input_tensor, [-1, vector_length])

    return vector


# 构建 full 全连接神经网络(nn)的每一层
# nodeNum: 该层的神经节点的总数
def fnn(input_tensor, nodeNum, activeFun=None):

    input_dim = int(input_tensor.get_shape()[-1])

    # 拼接出: 层的名称
    if activeFun!= None :
        layer_name ='fnn_'+ activeFun.func_name+'_'+ str(nodeNum)
    else:
        layer_name = 'fnn_linear_'+ str(nodeNum)

    print(layer_name)

    # 初始化 权重 w和b
    with tf.name_scope(layer_name):
        weight = tf.Variable(tf.random_normal([input_dim, nodeNum]), name='weight')
        bias = tf.Variable(tf.random_normal([nodeNum]), name='bias')

        if activeFun!= None :
            return activeFun(tf.add(tf.matmul(input_tensor, weight), bias))
        else:
            return tf.add(tf.matmul(input_tensor, weight), bias)


# Create model
def conv_net(x, dropout):
    # Reshape input picture
    with tf.name_scope('reshape'):
        x = tf.reshape(x, shape=[-1, 28, 28, 1])

    #输入x, 单通道图像, 本层的卷积核5*5, 共有32个核, 输出
    conv1 = cnn( x, 5, 32 )

    #输入conv1, 本层的卷积核5*5, 共有64核,输出
    conv2 = cnn( conv1, 5, 64)

    # 拉伸为一维向量
    fc1 = reshape_vector(conv2)

    # 第一层 全连接 1024个隐藏神经元节点
    nn1 = fnn(fc1, 1024, tf.nn.relu)

    # Apply Dropout
    nn1 = tf.nn.dropout(nn1, dropout)

    # 第二层, 10个分类节点
    nn2 = fnn(nn1, 10)

    return nn2
```

```python
# Construct model
pred = conv_net(x, keep_prob)

# Define loss and optimizer
# 观测 偏离
with tf.name_scope('Loss'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

# 调节参数 (随机梯度下降)
with tf.name_scope('SGD'):
    # 精度 0.98
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
with tf.name_scope('accuracy'):
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()
```

```python
cnn_28x28x1-5x5x32
cnn_14x14x32-5x5x64
[None, 7, 7, 64]
3136
fnn_relu_1024
fnn_linear_10
```

```python
# 写到目录
writer = tf.summary.FileWriter("../log1", tf.get_default_graph(), flush_secs=1)

writer.close()
```

```python
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc)
        step += 1
    print "Optimization Finished!"

    # Calculate accuracy for 256 mnist test images
    print "Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                      y: mnist.test.labels[:256],
                                      keep_prob: 1.})
```

```python
Iter 1280, Minibatch Loss= 15835.245117, Training Accuracy= 0.25781
Iter 2560, Minibatch Loss= 10163.646484, Training Accuracy= 0.49219
Iter 3840, Minibatch Loss= 7746.220703, Training Accuracy= 0.53906


Iter 198400, Minibatch Loss= 127.029541, Training Accuracy= 0.96875
Iter 199680, Minibatch Loss= 150.448425, Training Accuracy= 0.96875
Optimization Finished!
Testing Accuracy: 0.980469
```


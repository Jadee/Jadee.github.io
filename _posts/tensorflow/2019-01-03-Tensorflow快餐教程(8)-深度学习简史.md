---
title: Tensorflow快餐教程(8)-深度学习简史
date: 2019-01-03
categories: Tensorflow
tags:
- Tensorflow
---

#深度学习简史

## 从机器学习流派说起
如果要给机器学习划分流派的话，初步划分可以分为『归纳学习』和『统计学习』两大类。所谓『归纳学习』，就跟我们平时学习所用的归纳法差不多，也叫『从样例中学习』。

<!-- more -->

归纳学习又分为两大类，一类是像我们归纳知识点一样，把知识分解成一个一个的点，然后进行学习。因为最终都要表示成符号，所以也叫做『符号主义学习』；另一类则另辟蹊径，不关心知识是啥，而是模拟人脑学习的过程，人脑咋学咱们就照着学。这类思路模拟人的神经系统，因为人的神经网络是连接在一起的，所以也叫『连接主义学习』。

『统计学习』，则是上世经90年代才兴起的新学派。是一种应用数学和统计学方法进行学习的新思路。就是我既不关心学习的内容，也不是模拟人脑，而主要关心统计概率。这是一种脱离了主观，基本全靠客观的方式。

## 连接主义学派

连接主义学派的初心是模拟人脑的学习方式。

我们先从生理课的知识说起，先看看人脑最基本的组成部分 - 神经元。

![avatar](http://qiniu.xdpie.com/2018-05-17-10-59-35.png)

如上图所示，
一个神经元由三个主要部分组成：中间是细胞体，细胞体周围有若干起接收信号作用的树突，还有一条长长的轴突，用于将信号传导给远处的其他细胞。

神经细胞收到所有树突传来的信号之后，细胞体会产生化学反应，决定是否通过轴突输出给其他细胞。

比如皮肤上的感觉细胞接受了刺激之后，将信号传给附近的神经细胞的树突。达到一定强度之后，神经细胞会通过轴突传递给下一个神经细胞，一直传递到大脑。大脑做出反应之后，再通过运动神经元的轴突去刺激肌肉去进行反应。

这其中值得一提的是赫布理论。这是加拿大心理学家赫布在1949年出版的《行为组织学》中提出的，其内容是：如果一个神经元B在另一个神经元A的轴突附近，并且受到了A的信号的激活，那么A或B之一就会产生相应的增长变化，使得这个连接被加强。

这一理论一直到51年以后的2000年，才由诺贝尔医学奖得主肯德尔的动物实验所证实。但是在被证实之前，各种无监督机器学习算法其实都是赫布规则的一个变种。在被证明之前，就被广泛使用了。

## M-P神经元模型

在赫布原理提出6年前的1943年，虽然这时候电子计算机还没有被发明出来，距离我们的伟大偶像阿兰.图灵研究出来『图灵机测试』也还有3年时间，有两位传奇人物麦卡洛可和皮茨就发表了用算法模拟神经网络的文章。那一年，少年天才皮茨只有20岁！皮茨同学是个苦出身，15岁因为父亲让他退学，他一怒之下离家出走。那时候，他已经读完了罗素的《数学原理》这样一本大学教材。罗素后来把皮茨推荐给了著名哲学家，维也纳学派的代表人物卡尔纳普。后面我们讲归纳学习和归纳逻辑时还会说过卡尔纳普。卡尔纳普就把自己的哲学著作《语言的逻辑句法》送给初中生皮茨看，结果皮茨不过一个月就看完了。于是卡尔纳普惊为天人，请皮茨到芝加哥大学。。。打扫厕所！

后来，医生兼神经科学家麦卡洛可研究神经学需要一个懂数学的合作者，于是就选了17岁的清洁工皮茨。后来他们成为控制论创始人维纳的学生。后来因为被造谣陷害，皮茨跟维纳闹翻，46岁就英年早逝了。

神经网络的基础至今仍然是麦卡洛可和皮茨提出的模型，简称M-P模型。

## 感知机 - 人工神经网络的第一次高潮和低谷

1954年，IBM推出了IBM704计算机，并且有Fortran这样的算法语言。4年后， 1958年，康奈尔大学实验心理学家弗兰克.罗森布拉特根据M-P模型实现了第一个人工神经网络模型-感知机。

感知机的提出，使人类有了第一种可以模拟人脑神经活动的模型，迅速引起轰动。迎来了人工神经网络的第一个高潮。

感知机的模型如下图所示：

![avatar](http://5b0988e595225.cdn.sohucs.com/images/20171105/5787e6a5e1f34758b2a683fd0f5398bd.jpeg)

[感知机模型---思路图解](https://blog.csdn.net/yeziand01/article/details/89424907)

感知机由三部分组成：  
1. 输入：包括信号的输入强度和权值  
2. 求和：将输入求和  
3. 激活函数：根据求和的结果决定输出的值。

感知机了不起的地方在于，不需要任何先验知识，只要能够用一条直线把要解决的问题分为两部分，就可以区分。这种问题叫做线性可分问题。比如一些建筑在长安街以北，一些在长安街以南，感知机就能做到把这两部分建筑分开，尽管感知器根本不知道长安街是什么，东南西北是什么。

![avatar](http://image.bubuko.com/info/201909/20190928155234849980.png)

如上图所示，因为x和o可以找到一条直线分隔，所以感知机模型可以解决它。

而像下面这样红蓝点没法用一条直接分开的，就没办法应用感知机来区分它。

![avatar](https://graph.baidu.com/thumb/3368527302,249220879.jpg)

罗森布拉特比起少年扫地僧皮茨，可是名校高材生。他所就读的纽约Bronx科学高中，光诺贝尔奖获得者就有8个，此外还有6个普利策奖。也是这所学校，比他大一届的学长有个叫马文.明斯基的，是人工智能的奠基人之一。

正值感知器如日中天时，明斯基出版了著名的《感知机》一书，证明感知机连异或这种最基本的逻辑运算都无法解决。因为异或问题不是线性可分的，需要两条直线才可以，所以感知机模型确实解决不了。这一致命一击，使得人工神经网络的第一次高潮迅速被打入低谷。

不过，值得一提的是。后来深度学习的发展，跟模拟人的大脑越来越无关。学界认为不应该再叫『人工神经网络』，不妨就叫多层感知机MLP好了。

## 人工神经网络第二次高潮和低谷

单独的感知机无法解决的问题，是不是将多个感知机组合在一起就可以了呢？是的。1974年，哈佛大学学生保罗.沃波斯的博士论文提出了反向传播算法（简称BP算法），成功地解决了感知机不能实现异或的问题。实现的方法也基本上就是，一条直线不够，再加一条就好了。

但是，当时正是人工神经网络的第一次低谷中，纵然你是哈佛大学高材生也无人问津。这一重要成果当时没有造成大的影响。

在沃波斯的论文发表后10年后的1984年，这一年乔布斯推出了著名的苹果第一代mac电脑，加州理工学院的物理学家霍普菲尔德实现了他于两年前提出的一种循环神经网络模型。这个重要成果重新激发了大家对于人工神经网络的热情。

两年后的1986年，处于第二次人工神经网络热潮的学界再次重新发现了沃波斯提出过的BP算法。这更加促进了人工神经网络的发展。

感知器的局限在于它只有两层小网络。而BP算法给创造更多层更大型的网络创造了可能。  
BP算法的基本思想是：1.信号正向传播。2.误差反向传播给上层的每一个神经元。

我们在第一讲构建过的无脑全连接网络，就是这个时代的技术。我们再复习一下：
```python
# 所有连接随机生成权值
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w_h, w_o):
    h = tf.nn.sigmoid(tf.matmul(X, w_h)) 
    return tf.matmul(h, w_o) 
```
这些跟人工神经网络相关的函数，定义在tf.nn模块中，包括激活函数和卷积等功能。

通过BP算法，成功将神经网络做到了5层。然而，在超过5层时，遇到了困难。这个困难，困扰了研究者整整20年。  
这个困难主要有两方面，第一方面，随着层数的增多，反馈的误差对上层的影响越来越小。第二方面，层数增加之后，很容易被训练到一个局部最优值，而无法继续下去。

遇到了这个困难之后，大部分研究人员转而研究如何在少的层次上有所突破。正如我们前面所讲的，机器学习的另一大流派『统计学习』正是在这个时代取得了突破性的进展，其代表作是『支持向量机』-SVM。

## 深度学习时代

但是还是有极少数的研究人员在人工神经网络的第二次低潮中继续坐冷板凳研究。20年后的2006年，加拿大学者杰弗里.辛顿（Hinton）提出了有效解决多层神经网络的训练方法。他的方法是将每一层都看成一个无监督学习的受限玻尔兹曼机进行预训练提取特征，然后再采用BP算法进行训练。

这样的话，这些受限玻尔兹曼机就可以像搭积木一样搭得很高。这些由受限玻尔兹曼机搭起的网络叫做深度信念网络或者叫深层信念网络。这种采用深度信念网络的模型后来就叫做『深度学习』。

当然，Hinton也并不是在孤军奋战。他有位博士后高徒叫Yann Lecun。1989年，BP算法重新发现后的第3年，Lecun将BP算法成功应用在卷积神经网络CNN中。1998年，经过十年努力，Yann Lecun发明了LeNet。但是请注意这个时间点，这时候还没到2006年Hinton改变世界的时候，机器学习的王者是支持向量机SVM。

但是，机遇是留给有准备的人的。一方面CNN中的关键技术点ReLU和Dropout不断被解决；另一方面大数据和云计算引发的计算能力的突破，使得CNN可以使用更强大的计算能力来完成以前无法想象的任务。

我们在第一讲曾经讲过将简单一个隐藏层的全连接网络使用ReLU和Dropout技术的例子：
```python
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden): 
    X = tf.nn.dropout(X, p_keep_input)
    h = tf.nn.relu(tf.matmul(X, w_h))

    h = tf.nn.dropout(h, p_keep_hidden)
    h2 = tf.nn.relu(tf.matmul(h, w_h2))

    h2 = tf.nn.dropout(h2, p_keep_hidden)

    return tf.matmul(h2, w_o)
```

Tensorflow在tf.nn模块中为我们封装好了ReLU和Dropout，直接调用就行。

2012年，还是创造奇迹的Hinton和他的学生Alex Krizhevsky，在LeNet基础上改进的AlexNet一举夺取ImageNet图像分类的冠军，刷新了世界记录。促使卷积神经网络成为处理图像最有力的武器。

AlexNet之所以有这样大的进步，其主要原因有四种：
1. 为了防止过拟合，使用了Dropout和数据增强技术  
2. 采用了非线性激活函数ReLU  
3. 大数据量训练（大数据时代的作用！）  
4. GPU训练加速（硬件的进步）

下图是Alex网络的结构：

![avatar](http://picture.piggygaga.top/AlexNet/AlexNet.png)

我们看下Tensorflow中对于AlexNet的参考实现的删节版：
```python
def inference(images):
  parameters = []
  # conv1
  with tf.name_scope('conv1') as scope:
    kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]

  # lrn1
  with tf.name_scope('lrn1') as scope:
    lrn1 = tf.nn.local_response_normalization(conv1,
                                              alpha=1e-4,
                                              beta=0.75,
                                              depth_radius=2,
                                              bias=2.0)

  # pool1
  pool1 = tf.nn.max_pool(lrn1,
                         ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1],
                         padding='VALID',
                         name='pool1')

  # conv2
  with tf.name_scope('conv2') as scope:
    kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]

  # lrn2
  with tf.name_scope('lrn2') as scope:
    lrn2 = tf.nn.local_response_normalization(conv2,
                                              alpha=1e-4,
                                              beta=0.75,
                                              depth_radius=2,
                                              bias=2.0)

  # pool2
  pool2 = tf.nn.max_pool(lrn2,
                         ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1],
                         padding='VALID',
                         name='pool2')

  # conv3
  with tf.name_scope('conv3') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]

  # conv4
  with tf.name_scope('conv4') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv4 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]

  # conv5
  with tf.name_scope('conv5') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv5 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]

  # pool5
  pool5 = tf.nn.max_pool(conv5,
                         ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1],
                         padding='VALID',
                         name='pool5')

  return pool5, parameters
```

卷积神经网络是一种权值共享的网络，这个特点使其模型的复杂度显著降低。  
那么什么是卷积呢？卷积是泛函分析中的一种积分变换的数学方法，通过两个函数来生成第三个函数，表征两个函数经过翻转和平移的重叠部分的面积。  
在传统认别算法中，我们需要对输入的数据进行特征提取和数据重建，而卷积神经网络可以直接将图片做为网络的输入，自动提取特征。它的优越特征在于对于图片的平移、比例缩放、倾斜等变形有非常好的适应性。这种技术简直就是为了图形和语音而生的。从此，图片是正着放还是倒着放或者随便换个角度，远点还是近点等再也不是问题，使得识别率一下子显著提升到了可用的程度。

DBN和CNN双剑合壁，成功引发了图像和语音两个领域的革命。使得图像识别和语音识别技术迅速换代。  
不过问题还有一个，自然语言处理和机器翻译。这也是个老大难问题了，我们单纯想想就知道难度有多高。江山代有才人出，当Yann LeCun发表他那篇著名的论文时，文章第三作者叫做Yoshua Bengio。在神经网络低潮的90年代，Hinton研究DBN，LeCun研究CNN的时候，Yoshua在研究循环神经网络RNN，并且开启了神经网络研究自然语言处理的先河。后来，RNN的改进模型长短期记忆模型LSTM成功解决了RNN梯度消失的问题，从此成为自然语言处理和机器翻译中的利器。

Hinton, Yann LeCun和Yoshua Bengio，就是被国人称为『深度学习三巨头』的三位传奇人物。他们共同在神经网络第二次低潮的寒冬中，坚持自己所坚信的方向，最终一起改变了世界。

# 深度学习小结

对于多层感知机模型来说，深度学习时代仍然沿用了BP网络时代发明的反向传播算法和梯度下降技术。

CNN和RNN是目前使用最广泛的两项深度学习工具。

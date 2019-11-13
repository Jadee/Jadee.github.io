---
title: Tensorflow快餐教程(11)-API调用
date: 2019-01-03
categories: Tensorflow
tags:
- Tensorflow
---

# 高层封装API

在前面已经快速将CNN, RNN的大致概念和深度学习的简史走马观花过了一遍之后，我们就可以开始尝试使用高层封装的API。

<!-- more -->

## 模型 - 训练 - 评估 三条语句搞定

既然高层封装，我们就采用最简单的方式：首先是一个模型，然后就开始训练，最后评估一下效果如何。

我们还是举祖传的MNIST的例子。

核心三条语句，一句模型，一句训练，一句评估：
```python
estimator = tf.estimator.LinearClassifier(feature_columns=[image_column], n_classes=10)

# Train.
estimator.train(input_fn=train_input_fn, steps=2000)

# Evaluate and report metrics.
eval_metrics = estimator.evaluate(input_fn=eval_input_fn, steps=1)
```
我们首先知道MNIST是把手写图像分成十类，那么就用个线性回归分类器，指定分成10类：
```python
estimator = tf.estimator.LinearClassifier(feature_columns=[image_column], n_classes=10)
```
训练也是无脑的，指定训练多少步就是了：
```python
estimator.train(input_fn=train_input_fn, steps=2000)
```
评估也不需要懂啥，给个测试集就是了：
```python
eval_metrics = estimator.evaluate(input_fn=eval_input_fn, steps=1)
```
给大家一个完整能运行的例子，主要的工作量都在处理输入数据上，真正有功能的就是那三条语句：
```python
import numpy as np
import tensorflow as tf

def get_input_fn(dataset_split, batch_size, capacity=10000, min_after_dequeue=3000):

  def _input_fn():
    images_batch, labels_batch = tf.train.shuffle_batch(
        tensors=[dataset_split.images, dataset_split.labels.astype(np.int32)],
        batch_size=batch_size,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue,
        enqueue_many=True,
        num_threads=4)
    features_map = {'images': images_batch}
    return features_map, labels_batch

  return _input_fn

data = tf.contrib.learn.datasets.mnist.load_mnist()

train_input_fn = get_input_fn(data.train, batch_size=256)
eval_input_fn = get_input_fn(data.validation, batch_size=5000)

# Specify the feature(s) to be used by the estimator.
image_column = tf.contrib.layers.real_valued_column('images', dimension=784)
estimator = tf.estimator.LinearClassifier(feature_columns=[image_column], n_classes=10)

# Train.
estimator.train(input_fn=train_input_fn, steps=2000)

# Evaluate and report metrics.
eval_metrics = estimator.evaluate(input_fn=eval_input_fn, steps=1)
print(eval_metrics)
```

## 三步法进阶

现在我们已经学会三步法了。虽然不涉及底层细节，我们还是有很多工具可以做得更好的。

比如我们要自己设计优化方法, 从三条语句变成四条：
```python
optimizer2 = tf.train.FtrlOptimizer(learning_rate=5.0, l2_regularization_strength=1.0)
estimator2 = tf.estimator.LinearClassifier(
    feature_columns=[image_column], n_classes=10, optimizer=optimizer2)

# Train.
estimator2.train(input_fn=train_input_fn, steps=2000)

# Evaluate and report metrics.
eval_metrics2 = estimator2.evaluate(input_fn=eval_input_fn, steps=1)
print(eval_metrics2)
```
这段代码不是片断，拼接到上面的代码的后面就可以直接运行。

## 更进一步：支持向量机

默认的虽然通用，但是效果可能不如更专业的更好。比如我们想用前深度学习时代最强大的工具之一 - 支持向量机来进行MNIST识别。我们还是可以用高层API来实现。将LinearClassifier换成KernelLinearClassifier。
```python
optimizer3 = tf.train.FtrlOptimizer(
   learning_rate=50.0, l2_regularization_strength=0.001)

kernel_mapper3 = tf.contrib.kernel_methods.RandomFourierFeatureMapper(
  input_dim=784, output_dim=2000, stddev=5.0, name='rffm')
kernel_mappers3 = {image_column: [kernel_mapper3]}
estimator3 = tf.contrib.kernel_methods.KernelLinearClassifier(
   n_classes=10, optimizer=optimizer3, kernel_mappers=kernel_mappers3)

# Train.
estimator3.fit(input_fn=train_input_fn, steps=2000)

# Evaluate and report metrics.
eval_metrics3 = estimator3.evaluate(input_fn=eval_input_fn, steps=1)
print(eval_metrics3)
```
我们来比较一下三种方法：
```python
Elapsed time: 80.69186925888062 seconds
{'loss': 0.26811677, 'accuracy': 0.9228, 'global_step': 2000}
Elapsed time: 80.33205699920654 seconds
{'loss': 0.26356304, 'accuracy': 0.9276, 'global_step': 2000}
Elapsed time: 98.87778902053833 seconds
{'loss': 0.10834637, 'accuracy': 0.9668, 'global_step': 2000}
```
SVM支持向量机力量果然强大，从92%的识别率提升到了96%.

## 高层深度学习API

准备数据的语句不变，我们再加一种采用深度学习的方式，也是三步：
```python
classifier = tf.estimator.DNNClassifier(
    feature_columns=[image_column],
    hidden_units=[784, 625],
    n_classes=10)

# Train.
classifier.train(
    input_fn=train_input_fn,
    steps=2000)

eval_result = classifier.evaluate(
    input_fn=eval_input_fn, steps=1)

print(eval_result)
```
打印出来的结果如下：
```python
{'accuracy': 0.9812, 'average_loss': 0.064692736, 'loss': 323.46368, 'global_step': 2000}
```
识别率达到98%，比支持向量机还要强一些。

## Tensorflow的API结构

![avatar](http://img.wandouip.com/1872659-ab2cce108cee949130a2015972ec5e50.png)

我们从第一讲到第十讲学习的都是Mid-Level API。这一讲讲的是High-Level API。

**Tensorflow r1.8 Estimators API的变化**

Tensorflow API的变化一向以迅速著称，兼容性也不是很好。

tf.estimator.Estimators的前身是tf.contrib.learn.Estimators。

我们对比一下LinearClassifier在这两个版本的区别：

新版：
```python
estimator = tf.estimator.LinearClassifier(feature_columns=[image_column],
                                          n_classes=10)

# Train.
estimator.train(input_fn=train_input_fn, steps=2000)

# Evaluate and report metrics.
eval_metrics = estimator.evaluate(input_fn=eval_input_fn, steps=1)
```
旧版：
```python
estimator = tf.contrib.learn.LinearClassifier(feature_columns=[image_column], n_classes=10)

# Train.
estimator.fit(input_fn=train_input_fn, steps=2000)

# Evaluate and report metrics.
eval_metrics = estimator.evaluate(input_fn=eval_input_fn, steps=1)
print(eval_metrics)
```
主要区别为：  
1. 包名改变了  
2. 新版的训练方法是train，而旧版是fit。

因为新版本没有提供支持向量机的分类器，我们用的核函数版本的KernelLinearClassifier还是老的包中的，所以还是用的fit来训练。

# 参考

1. <https://www.wandouip.com/t5i2925/>


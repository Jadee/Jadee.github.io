---
title: Tensorflow快餐教程(3)-向量
date: 2019-01-03
categories: Tensorflow
tags:
- Tensorflow
---

# 向量

向量在编程语言中就是最常用的一维数组。二维数组叫做矩阵，三维以上叫做张量。

<!-- more -->

向量虽然简单，高效，且容易理解。但是与操作0维的标量数据毕竟还是不同的。比如向量经常用于表示一个序列，生成序列像标量一样一个一个手工写就不划算了。当然可以用循环来写。在向量中这样还好，如果是在矩阵或者是张量中就强烈建议不要用循环来做了。系统提供的函数一般都是经过高度优化的，而且可以使用GPU资源来进行加速。我们一方面尽可能地多使用系统的函数，另一方面也不要迷信它们，代码优化是一个实践的过程，可以实际比较测量一下。

# 快速生成向量的方法

## range函数生成等差数列

tf.range函数用来快速生成一个等差数列。相当于之前我们讲numpy时的np.arange函数。

原型：
```python
tf.range(start, limit, delta=1, dtype=None, name='range')
```

例：
```python
>>> b11 = tf.range(1,100,1)
>>> b11
<tf.Tensor 'range:0' shape=(99,) dtype=int32>
>>> sess.run(b11)
array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
       18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
       35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
       52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
       69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85,
       86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99],
      dtype=int32)
```

## linspace生成浮点等差数组

tf.linspace与tf.range的区别在于，数据类型不同。
```
tf.lin_space(
    start,
    stop,
    num,
    name=None
)
```
其中，start和stop必须是浮点数，且类型必须相同。num必须是整数。

例：
```python
>>> a2 = tf.linspace(1.0,10.0,4)  
>>> a2
<tf.Tensor 'LinSpace_2:0' shape=(4,) dtype=float32>
>>> sess.run(a2)
array([ 1.,  4.,  7., 10.], dtype=float32)
```

## 拼瓷砖

就是将一段向量重复若干次。
```python
>>> a10 = tf.range(1,4,1)
>>> sess.run(a10)
array([1, 2, 3], dtype=int32)
>>> a11 = tf.tile(a10,[3])
>>> sess.run(a11)
array([1, 2, 3, 1, 2, 3, 1, 2, 3], dtype=int32)
```

# 向量操作

## 将向量反序

可以使用tf.reverse函数。  
原型：
```
tf.reverse(
    tensor,
    axis,
    name=None
)
```
tensor是向量，axis轴对于向量不重要，给个[-1]就可以了。折腾轴是张量时间的事情，暂时还用不到。
```python
>>> a2 = tf.linspace(1.0,10.0,4)
>>> a3 = tf.reverse(a2,[-1])
>>> sess.run(a3)
array([10.,  7.,  4.,  1.], dtype=float32)
```
## 切片

切片也是向量的常用操作之一，就是取数组的一部分。

例：
```python
>>> a5 = tf.linspace(1.0,100.0, 10)
>>> sess.run(a5)
array([  1.,  12.,  23.,  34.,  45.,  56.,  67.,  78.,  89., 100.],
      dtype=float32)
>>> a6 = tf.slice(a5, [2],[4])
>>> sess.run(a6)
array([23., 34., 45., 56.], dtype=float32)
```
将来处理张量时，我们从一个矩阵切一块，或从一个张量中切一块，就好玩得多了。但是原理跟向量上是一样的。

## 连接

tf.concat也是需要给定轴信息的。对于两个线性的向量，我们给0或者-1就好。
```python
>>> a20 = tf.linspace(1.0,2.0,10)
>>> sess.run(a20)
array([1.       , 1.1111112, 1.2222222, 1.3333334, 1.4444444, 1.5555556,
       1.6666667, 1.7777778, 1.8888888, 2.       ], dtype=float32)
>>> a21 = tf.linspace(2.0,3.0,5)
>>> sess.run(a22)
array([1.       , 1.1111112, 1.2222222, 1.3333334, 1.4444444, 1.5555556,
       1.6666667, 1.7777778, 1.8888888, 2.       , 2.       , 2.25     ,
       2.5      , 2.75     , 3.       ], dtype=float32)
>>> a23 = tf.concat([a20,a21],-1)
>>> sess.run(a23)
array([1.       , 1.1111112, 1.2222222, 1.3333334, 1.4444444, 1.5555556,
       1.6666667, 1.7777778, 1.8888888, 2.       , 2.       , 2.25     ,
       2.5      , 2.75     , 3.       ], dtype=float32)
```

# 向量计算

## 向量加减法
同样长度的向量之间可以进行加减操作。  
例：
```python
>>> a40 = tf.constant([1,1])
>>> a41 = tf.constant([2,2])
>>> a42 = a40 + a41
>>> sess.run(a42)
array([3, 3], dtype=int32)
>>> a43 = a40 - a41
>>> sess.run(a43)
array([-1, -1], dtype=int32)
>>> a43
<tf.Tensor 'sub:0' shape=(2,) dtype=int32>
```
## 向量乘除标量

向量乘除标量也非常好理解，就是针对向量中的每个数都做乘除法。  
例：
```python
>>> a44 = a40 * 2
>>> sess.run(a44)
array([2, 2], dtype=int32)
>>> a45 = a44 / 2  
>>> sess.run(a45)
array([1., 1.])
>>> a44
<tf.Tensor 'mul:0' shape=(2,) dtype=int32>
>>> a45
<tf.Tensor 'truediv_1:0' shape=(2,) dtype=float64>
```

## 广播运算

如果针对向量和标量进行加减运算，也是会对向量中的每个数进行加减运算。这种操作称为广播操作。  
例：
```python
>>> a46 = a40 + 1
>>> sess.run(a46)
array([2, 2], dtype=int32)
>>> a46
<tf.Tensor 'add_1:0' shape=(2,) dtype=int32>
```
## 向量乘法

两个向量相乘，默认的运算是求元素对应乘积（element-wise product)，也叫做Hadamard积。  
例：
```python
>>> b1 = tf.constant([1,2])
>>> b2 = tf.constant([2,1])
>>> b3 = b1 * b2
>>> b3
<tf.Tensor 'mul_7:0' shape=(2,) dtype=int32>
>>> sess.run(b3)
array([2, 2], dtype=int32)
```
直接调用tf.multiply也是同样的效果，例：
```python
>>> b4 = tf.multiply(b1,b2)
>>> b4   
<tf.Tensor 'Mul_2:0' shape=(2,) dtype=int32>
>>> sess.run(b4)
array([2, 2], dtype=int32)
```

如果要计算点积(dot product)的话，我们得提前剧透一下矩阵的内容了。

首先，用向量是没法做矩阵计算的。  
例：
```python
>>> a21 = tf.constant([2,3]) 
>>> a22 = tf.constant([4,5])
>>> a21   
<tf.Tensor 'Const_20:0' shape=(2,) dtype=int32>
>>> a22
<tf.Tensor 'Const_21:0' shape=(2,) dtype=int32>
```
这样(2,)的形状是向量，我们得先把它转换成(2,1)这样的单行矩阵，如下：
```python
>>> a31 = tf.constant(sess.run(tf.reshape(a21,[2,1])))
>>> a32 = tf.constant(sess.run(tf.reshape(a22,[2,1])))
>>> a31
<tf.Tensor 'Const_22:0' shape=(2, 1) dtype=int32>
>>> a32
<tf.Tensor 'Const_23:0' shape=(2, 1) dtype=int32>
```
下面我们终于可以计算点积了，我们知道点积A.B相当于A的转置乘以B，我们可以通过matmul函数来进行矩阵乘法。
```python
>>> a31 = tf.matmul(a31,a32,transpose_a=True)
>>> sess.run(a31)
array([[23]], dtype=int32)
```

我们也可以用tf.tensordot函数来计算点积。我们刚才为什么没用呢？答案是tensordot要求是浮点型矩阵。  
例：

第一步，需要浮点数：
```python
>>> f01 = tf.constant([1,1],dtype=tf.float32) 
>>> f02 = tf.constant([1,2],dtype=tf.float32)
```
第二步，reshape成单行矩阵：

```python
>>> f11 = tf.constant(sess.run(tf.reshape(f01,[2,1])))
>>> f12 = tf.constant(sess.run(tf.reshape(f02,[2,1])))
>>> f11
<tf.Tensor 'Const_26:0' shape=(2, 1) dtype=float32>
>>> f12
<tf.Tensor 'Const_27:0' shape=(2, 1) dtype=float32>
```
第三步，调用tensordot

```python
>>> f13 = tf.tensordot(f11,f12,2)
>>> sess.run(f13)
3.0
```

# 小结

从上面我们学习的函数我们可以看到，与普通语言中提供的函数多是为一维数组操作不同，Tensorflow中的切片、拼接等操作也是基于张量的。当我们后面学到张量遇到困难时，不妨回来看下这一节。不管后面张量多么复杂，其实也只是从一维向二维和多维推广而己。





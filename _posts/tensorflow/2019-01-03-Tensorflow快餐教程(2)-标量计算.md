---
title: Tensorflow快餐教程(2)-标量计算
date: 2019-01-03
categories: Tensorflow
tags:
- Tensorflow
---

Tensorflow的Tensor意为张量。一般如果是0维的数组，就是一个数据，我们称之为标是Scalar；1维的数组，称为向量Vector；2维的数组，称为矩阵Matrics；3维及以上的数组，称为张量Tensor。

<!-- more -->

在机器学习中，用途最广泛的是向量和矩阵的运算。这也是我们学习中的第一个难关。这一节我们先打标量的基础。

上节我们学过，Tensorflow的运行需要一个Session对象。下面代码中所用的sess都是通过
```python
sess = tf.Session()
```
获取的Session对象，以下就都省略不写了。

# 标量Scalar

标量是指只有一个数字的结构。尝试将一个整数赋给一个Tensorflow的常量，看看是什么效果：
```python
>>> a10 = 1
>>> b10 = tf.constant(a10)
>>> print(b10)
Tensor("Const_6:0", shape=(), dtype=int32)
>>> sess.run(b10)
1
```

我们可以看到，tf.constant(a10)生成了一个shape为空的，类型为int32的张量。

Tensorflow是一个经过数据类型优化的高性能系统，所以对于数据类型的要求比较高。  
比如我们想对上面的标量b10进行求正弦值的运算，就会得到下面的错误，sin运算只支持浮点数和复数类型：

```python
>>> b11 = tf.sin(b10)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/usr/lib/python3.6/site-packages/tensorflow/python/ops/gen_math_ops.py", line 6862, in sin
    "Sin", x=x, name=name)
  File "/usr/lib/python3.6/site-packages/tensorflow/python/framework/op_def_library.py", line 609, in _apply_op_helper
    param_name=input_name)
  File "/usr/lib/python3.6/site-packages/tensorflow/python/framework/op_def_library.py", line 60, in _SatisfiesTypeConstraint
    ", ".join(dtypes.as_dtype(x).name for x in allowed_list)))
TypeError: Value passed to parameter 'x' has DataType int32 not in list of allowed values: float16, bfloat16, float32, float64, complex64, complex128
```
后面我们还会多次遇到数据类型不符合要求，以至于无法运算的错误。
所以我们首先要学习下Tensorflow的数据类型。

## Tensorflow的数据类型

Tensorflow主要支持以下数据类型

* 整型：  
  * tf.int8: 8位带符号整数  
  * tf.uint8: 8位无符号整数  
  * tf.int16: 16位带符号整数  
  * tf.int32: 32位带符号整数  
  * tf.int64: 64位带符号整数  
  
* 浮点型：  
  * tf.float32: 32位浮点数  
  * tf.float64: 64位浮点数
  
* 复数:
  * tf.complex64: 64位复数  
  * tf.complex128: 128位复数
  
在Tensorflow的很多运算中，都支持通过dtype=的方式来指定数据类型。  
例：
```python
>>> b01 = tf.constant(1,dtype=tf.uint8)
>>> print(b01)
Tensor("Const_7:0", shape=(), dtype=uint8)
>>> b02 = tf.constant(1,dtype=tf.float64)
>>> print(b02)
Tensor("Const_8:0", shape=(), dtype=float64)
>>> sess.run(b01)
1
>>> sess.run(b02)
1.0
```

## Tensor到某类型数据的转换

通过tf.constant函数，我们可以将数据转换成Tensor。同样，Tensorflow也提供了Tensor到各种数据类型的转换函数。  
例，将Tensor转换成tf.int32:
```python
>>> b03 = tf.to_int32(b02)
>>> print(b03)
Tensor("ToInt32:0", shape=(), dtype=int32)
>>> sess.run(b03)
1
>>> b04 = sess.run(b03)
>>> print(b04)
1
```
从上面代码可以看到，b03 run的结果就是一个整数，不是Tensor。类似的函数还有tf.to_int64, tf.to_float, tf.to_double等。

定义这么多函数太麻烦了，还有一个通用的转换函数tf.cast. 格式为：tf.cast(Tensor, 类型名)。
例：
```python
>>> b05 = tf.cast(b02, tf.complex128)
>>> sess.run(b05)
(1+0j)
```

## 饱和转换

如果是将大类型如int64转成小类型int16，tf.cast转换可能会产生溢出。这在机器学习的计算中是件可怕的事情。在这种情况下，我们就需要使用饱和类型转换saturate_cast来保驾护航。

比如我们要把65536转换成tf.int8类型：
```python
>>> b06 = tf.constant(65536,dtype=tf.int64)
>>> print(b06)
Tensor("Const_9:0", shape=(), dtype=int64)
>>> sess.run(b06)
65536
>>> b07 = tf.saturate_cast(b06,tf.int8)
>>> sess.run(b07)
127
```
## 标量算术运算

标量Tensor常量可以进行算术运算。本质上是调用tf.add, tf.sub, tf.mul, tf.truediv, tf.mod等重载函数。  
例：
```python
>>> d01 = tf.constant(1)
>>> d02 = tf.constant(2)
>>> d_add = d01 + d02
>>> print(d_add)
Tensor("add:0", shape=(), dtype=int32)
>>> d_sub = d01 - d02
>>> print(d_sub)
Tensor("sub:0", shape=(), dtype=int32)
>>> d_mul = d01 * d02
>>> print(d_mul)
Tensor("mul:0", shape=(), dtype=int32)
>>> d_div = d01 / d02
>>> print(d_div)
Tensor("truediv:0", shape=(), dtype=float64)
>>> d_mod = d01 % d02
>>> print(d_mod)
Tensor("mod:0", shape=(), dtype=int32)
>>> d_minus = -d01
>>> print(d_minus)
Tensor("Neg:0", shape=(), dtype=int32)
```

对于除法多说两句，Tensor有两种除法，一种是"/"，另一种是"//"。"/"是浮点除法，对应的是tf.truediv，而"//"是计算整除，对应tf.floordiv。
```python
>>> d_div = d01 / d02
>>> print(d_div)
Tensor("truediv:0", shape=(), dtype=float64)
>>> d_div2 = d01 // d02
>>> print(d_div2)
Tensor("floordiv:0", shape=(), dtype=int32)
```

## 标量逻辑运算

对于>, <, >=, <=等关系，都会生成一个需要Session来运算的Tensor对象。只有==是例外，它会立即返回这两个Tensor是否是同一对象的结果。
```python
>>> d11 = d01 > d02
>>> d12 = d01 < d02
>>> d13 = d01 == d02
>>> d14 = d01 >= d02
>>> d15 = d01 <= d02
>>> print(d11)
Tensor("Greater_1:0", shape=(), dtype=bool)
>>> print(d12)
Tensor("Less:0", shape=(), dtype=bool)
>>> print(d13)
False
>>> print(d14)
Tensor("GreaterEqual:0", shape=(), dtype=bool)
>>> print(d15)
Tensor("LessEqual:0", shape=(), dtype=bool)
>>> d11 = d01 > d02
```

## 常用标量数学函数

首先还是强调一下注意类型，比如整形，一定要先转换成浮点型才能进行sqrt，sin等数学函数计算。  
例：
```python
>>> d31 = tf.constant(100, dtype=tf.float64)
>>> d32 = tf.sqrt(d31)
>>> sess.run(d32)
10.0
```
另外不要忘了，像sin, cos, tan这些函数是支持复数的哦。
例：
```python
>>> d40 = tf.constant(1+2j)
>>> d41 = tf.sin(d40)
>>> sess.run(d41)
(3.165778513216168+1.9596010414216063j)
```
中间结果也可以不用Tensor保存，直接用立即数，例：
```python
>>> d42 = tf.cos(0.5+0.3j)
>>> sess.run(d42)
(0.917370851271881-0.14599480570180629j)
```

# 常量、占位符和变量
前面我们主要使用立即数和常量。常量是通过tf.constant定义的，一旦定义就不能改变值的Tensor。如果要想改变Tensor的值，有两种变法：一种是根本就不赋值，先放个占位符；另一种是初始化成一个带值的变量，将来再改变值。
下面简单介绍一下占位符和变量。

## placeholder占位符

在算法计算时，有很多公式需要的数值是需要从外部拿到的，随时替换的。这时候我们就可以用一个占位符来写Tensor，需要计算时再把真数据通过feed_dict给填充进去就可以。  
我们来看个例子：
```python
>>> d50 = tf.placeholder(tf.float32, name ="input1")
>>> d51 = tf.sin(d50)
>>> sess.run(d51, feed_dict={d50: 0.2})
0.19866933
```
d50开始只用个placeholder，这样的话是没有办法通过之前不加feed_dict参数的sess.run来运行的。通过指定feed_dict={d50: 0.2}，我们就用数据替换掉了placeholder，就可以正常运行了。

## 变量
变量与占位符不同的一点是，变量在使用之前需要做初始化。初始化不但要在变量定义时写，还要调用相应的函数在使用前执行才可以。  
我们还是举例说明：
```python
>>> d60 = tf.Variable(1, dtype=tf.float32, name='number1')
>>> d61 = tf.tan(d60)
>>> init_op = tf.global_variables_initializer()
>>> sess.run(init_op)
>>> sess.run(d61)
1.5574077
```
在使用变量之前，我们可以一次性调用tf.global_variables_initializer函数去初始化所有变量，并且通过Session去执行。在此之后才能使用变量。

变量初始化之后，就可以通过assign函数来赋新值，例：
```python
>>> d62 = d60.assign(d60 * 2)
>>> sess.run(d62)
2.0
>>> sess.run(d61)
-2.1850398
```

# 小结

小结一下，这节主要介绍了数据类型，标量常用的计算函数，还有使用占位符和变量的方法。下一节我们正式开始线性代数之旅，走进向量、矩阵和张量。


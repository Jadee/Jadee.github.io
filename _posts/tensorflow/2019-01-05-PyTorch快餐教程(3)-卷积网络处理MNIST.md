---
title: PyTorch快餐教程(3)-卷积网络处理MNIST
date: 2019-01-05
categories: PyTorch
tags:
- PyTorch
---

# PyTorch卷积网络速度

我们还是先看下keras的卷积网络的写法：

<!-- more -->
```python
model = Sequential()
# 第1个卷积层，32个卷积核，大小为3*3，输入形状为(28,28,1)
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28,28,1)))
# 第2个卷积层，64个卷积核
model.add(Conv2D(64, (3, 3), activation='relu'))
# 第1个池化层
model.add(MaxPooling2D(pool_size=(2, 2)))
# 第1个Dropout层
model.add(Dropout(0.25))
# 卷积网络到全连接网络的转换层
model.add(Flatten())
# 第1个全连接层
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
# 输出层
model.add(Dense(10, activation='softmax'))
```

我们照方抓药，将其翻译成PyTorch。

Keras的Conv2D，在PyTorch中对应的是torch.nn.Conv2d。其他的也类似。

唯一的不同在于Flatten()这个层。

经过卷积计算以后，卷积的输出结果是一个4元的张量。分别是（元素数，深度，长，宽）。而全连接网络只需要一个（元素数，节点数）2元就可以了。

对于简便易学的Keras，Plattern层负责做这个转换，我们不需要了解经过每次卷积，输出变成了什么形状。但是对于PyTorch，这个转换是需要自己来写的，我们需要计算出输出的形状。

经过一计算，我们发现上面写keras代码时没有注意到的事情。就是卷积卷得不够狠。28*28的输入值，用3*3卷积，补上一圈padding，结果还是28*28。但是深度从1变成了16.

经过第二轮卷积加上最大池化之后，仍然有27*27之大。  
代码如下：
```python
# CNN网络
class FirstCnnNet(t.nn.Module):
    # 初始化传入输入、隐藏层、输出三个参数
    def __init__(self, num_classes):
        super(FirstCnnNet, self).__init__()
        # 输入深度1，输出深度16
        self.conv1 = t.nn.Sequential(
            t.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            t.nn.BatchNorm2d(16),
            t.nn.ReLU())
        self.conv2 = t.nn.Sequential(
            t.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            t.nn.BatchNorm2d(32),
            t.nn.ReLU(),
            t.nn.MaxPool2d(kernel_size=2, stride=1),
            t.nn.Dropout(p=0.25))
        self.dense1 = t.nn.Sequential(
            t.nn.Linear(27*27*32, 128),
            t.nn.ReLU(),
            t.nn.Dropout(p=0.25)
        )
        self.dense2 = t.nn.Linear(128,num_classes)

    # 传入计算值的函数，真正的计算在这里
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)
        x = self.dense1(x)
        x = self.dense2(x)
        return x
```
所以我们对卷积参数进行修改，加大步幅。第1次从28*28*1提取成14*14*16，第二次再提取成6*6*32。代码如下：
```python
# CNN网络
class FirstCnnNet(t.nn.Module):
    # 初始化传入输入、隐藏层、输出三个参数
    def __init__(self, num_classes):
        super(FirstCnnNet, self).__init__()
        # 输入深度1，输出深度16。从1,28,28压缩为16,14,14
        self.conv1 = t.nn.Sequential(
            t.nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            t.nn.BatchNorm2d(16),
            t.nn.ReLU())
        # 输入深度16，输出深度32。16,14,14压缩到32,6,6
        self.conv2 = t.nn.Sequential(
            t.nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            t.nn.BatchNorm2d(32),
            t.nn.ReLU(),
            t.nn.MaxPool2d(kernel_size=2, stride=1),
            t.nn.Dropout(p=0.25))
        # 第1个全连接层，输入6*6*32，输出128
        self.dense1 = t.nn.Sequential(
            t.nn.Linear(6*6*32, 128),
            t.nn.ReLU(),
            t.nn.Dropout(p=0.25)
        )
        # 第2个全连接层，输入128，输出10类
        self.dense2 = t.nn.Linear(128,num_classes)

    # 传入计算值的函数，真正的计算在这里
    def forward(self, x):
        x = self.conv1(x) # 16,14,14
        x = self.conv2(x) # 32,6,6
        x = x.view(x.size(0),-1)
        x = self.dense1(x) # 32*6*6 -> 128
        x = self.dense2(x) # 128 -> 10
        return x
```
经过上面的卷积代码之后，我们可以很快取得99%以上的准确率。

## 完整代码

完整代码如下，大家可以在此基础上修改下参数，体会下PyTorch编程的方法：
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
y_test = read_labels('./t10k-labels-idx1-ubyte', 10000)


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
num_epochs = 30
# 学习率
learning_rate = 1e-3
# 批量大小
batch_size = 100

# CNN网络
class FirstCnnNet(t.nn.Module):
    # 初始化传入输入、隐藏层、输出三个参数
    def __init__(self, num_classes):
        super(FirstCnnNet, self).__init__()
        # 输入深度1，输出深度16。从1,28,28压缩为16,14,14
        self.conv1 = t.nn.Sequential(
            t.nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            t.nn.BatchNorm2d(16),
            t.nn.ReLU())
        # 输入深度16，输出深度32。16,14,14压缩到32,6,6
        self.conv2 = t.nn.Sequential(
            t.nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            t.nn.BatchNorm2d(32),
            t.nn.ReLU(),
            t.nn.MaxPool2d(kernel_size=2, stride=1),
            t.nn.Dropout(p=0.25))
        # 第1个全连接层，输入6*6*32，输出128
        self.dense1 = t.nn.Sequential(
            t.nn.Linear(6*6*32, 128),
            t.nn.ReLU(),
            t.nn.Dropout(p=0.25)
        )
        # 第2个全连接层，输入128，输出10类
        self.dense2 = t.nn.Linear(128,num_classes)

    # 传入计算值的函数，真正的计算在这里
    def forward(self, x):
        x = self.conv1(x) # 16,14,14
        x = self.conv2(x) # 32,6,6
        x = x.view(x.size(0),-1)
        x = self.dense1(x) # 32*6*6 -> 128
        x = self.dense2(x) # 128 -> 10
        return x

# 输入28*28，隐藏层128，输出10类
model = FirstCnnNet(10)
print(model)

# 优化器仍然选随机梯度下降
optimizer = t.optim.Adam(model.parameters(), lr=learning_rate)

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
        X0 = X0.view(-1,1,28,28)
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

X_val = X_val.view(-1,1,28,28)

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


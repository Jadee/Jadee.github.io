---
title: Markov决策过程(mdp)
date: 2019-02-01
categories: 强化学习
tags:
- DNN
- 强化学习
- github
---

# Markov 决策过程(mdp) 解读
MDP 的ppt
http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/MDP.pdf

MDP 的 英文教程
https://ieor8100.github.io/rl/docs/Lecture%201%20-MDP.pdf

q-learning
https://www.zhihu.com/question/26408259

英文教程 q-learing
http://mnemstudio.org/path-finding-q-learning-tutorial.htm

q-learing 代码实现
https://github.com/JasonQSY/ML-Weekly/blob/master/P5-Reinforcement-Learning/Q-learning/Q-Learning-Get-Started.ipynb

# Markov 属性， 简化运算
Once the state is known, the history may be thrown away
向前看， 莫回头

# 无记忆的随机过程
A Markov process is a memoryless random process

# 状态转移概率矩阵
P is a state transition probability matrix,

Pss′ = P[St+1=s′|St=s]

![avatar](/images/RL-learning/mdp-1.png)
https://github.com/Jadee/Jadee.github.io/blob/master/images/RL-learning/mdp-1.png?raw=true

# 奖励 Reward 立即回报
举例：
大学state, 毕业action，立即回报Reward是：
一张毕业证(状态s=大学)+ 生命里耗费4年青春

硕士state, 毕业action，立即回报Reward是：
一张毕业证+ 生命里耗费3年青春-房价翻几倍了

A Markov Reward Process is a tuple ⟨S, P, R, γ⟩

s: 有限状态
p: 状态转移概率矩阵
无记忆

R: 奖励函数
Rs = E[Rt+1 | St = s]

γ： 折扣系数
对远期状态s'

# 回报 Return 长期待遇
Gain = Rt_1+ γRt_2 + γγRt_3

折扣系数 γ [0-1], 代表未来的奖励的折现

立即奖励 + 远期奖励的折现
immediate reward above delayed reward

# V(s) 值函数：
状态（职级) 对应的平均回报(待遇)
v(s)=E[Gt |St =s]

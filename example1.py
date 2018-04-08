# View more python tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
# from __future__ import print_function
import tensorflow as tf
import numpy as np

# create data
x_data = np.random.rand(100).astype(np.float32)  # datas
y_data = x_data*0.1 + 0.3  # result

### create tensorflow structure start ###

Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))  # 0.1，用随机数列方式生成参数变量
biases = tf.Variable(tf.zeros([1]))  # 0.3

y = x_data*Weights + biases

loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)  # 实现梯度下降算法的优化器
train = optimizer.minimize(loss)

# y：判别值
# y_dat：标签（真实值）
# loss：均方误差

### create tensorflow structure end ###

sess = tf.Session()
# 初始化变量，类似懒加载
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

# 开始训练
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))

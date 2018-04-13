import tensorflow as tf

# 这里通过numpy工具包生成模拟数据集
from numpy.random import RandomState

# 定义训练数据batch的大小
batch_size = 8

# 定义神经网络的参数，这里还是沿用3.4.2小节中给出的神经网络结构
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 在shape的一个维度上使用None可以方便使用不大的batch大小。
# 在训练时需要把数据分成比较小的batch，但是在测试时可以一次性使用全部数据。
# 当数据集较小时这样比较方便测试，但数据集较大时，将大量数据放入一个batch可以导致内存溢出。
x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

# 定义前向传播过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
y = tf.sigmoid(y)

# 定义损失函数及反向传播算法
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))
                                + (1 - y_) * tf.log(tf.clip_by_value(1 - y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# 生成模拟数据集。
rdm = RandomState(1)
X = rdm.rand(128, 2)
# 定义规则来给出样本的标签。在这里所有x1+x2<1的样例都被认为是正样本（比如零件合格）
# 而其他为负样本（比如零件不合格）。和TensorFlow游乐场中表示法不大一样的地方是，
# 在这里使用0来表示负样本，1来表示正样本。大部分解决分类问题的神经网络都会采用0和1的表示法
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

# 创建一个会话来运行TensorFlow程序。
with tf.Session() as sess:
    # 初始化变量
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # 输出目前（未经训练）的参数取值。
    print(sess.run(w1))
    print(sess.run(w2))
    print("\n")

    # 训练模型。
    STEPS = 5000
    for i in range(STEPS):
        # 每次取batch size个样本进行训练
        start = (i * batch_size) % 128
        end = (i * batch_size) % 128 + batch_size
        # 通过选取的样本训练神经网络并更新参数
        sess.run([train_step, y, y_], feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0:
            # 每1000步输出交叉熵
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print("After %d training step(s), cross entropy on all data is %g" % (i, total_cross_entropy))

    # 输出训练后的参数取值。
    print("\n")
    print(sess.run(w1))
    print(sess.run(w2))

    # 随着训练的进行，交叉熵是逐渐变小的。交叉熵越小说明预测结果与真实结果差距越小

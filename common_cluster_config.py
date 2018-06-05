import tensorflow as tf

# 定义两个工作。ps（parameter server）负责存储、获取以及更新变量的取值
# worker负责运行反向传播算法来获取参数梯度
tf.train.ClusterSpec({
    'worker': {
        'tf-workder0:2222',
        'tf-workder1:2222',
        'tf-workder2:2222',
    },
    'ps': {
        'tf-ps1:2222',
        'tf-ps2:2222',
    },
})

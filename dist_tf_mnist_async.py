# 计算图之间分布式（between-graph replication）异步模式样例程序
# coding=utf-8
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference

# 配置神经网络的参数。
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 1000
MOVING_AVERAGE_DECAY = 0.99

# 模型保存的路径和文件名。
MODEL_SAVE_PATH = "logs/log_async"
DATA_PATH = "mnist_data"

# 通过 flags 指定运行的参数。对于不同的任务（task）运行不同的程序。
FLAGS = tf.app.flags.FLAGS

# 指定当前程序是参数服务器还是计算服务器。
tf.app.flags.DEFINE_string('job_name', 'worker', ' "ps" or "worker" ')
# 指定集群中的参数服务器地址。
tf.app.flags.DEFINE_string(
    'ps_hosts', ' tf-ps0:2222,tf-ps1:1111',
    'Comma-separated list of hostname:port for the parameter server jobs. e.g. "tf-ps0:2222,tf-ps1:1111" ')
# 指定集群中的计算服务器地址。
tf.app.flags.DEFINE_string(
    'worker_hosts', ' tf-worker0:2222,tf-worker1:1111',
    'Comma-separated list of hostname:port for the worker jobs. e.g. "tf-worker0:2222,tf-worker1:1111" ')
# 指定当前程序的任务ID。
tf.app.flags.DEFINE_integer('task_id', 0, 'Task ID of the worker/replica running the training.')


# 定义TensorFlow的计算图，并返回每一轮迭代时需要运行的操作。
def build_model(x, y_, is_chief):
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    # 通过和5.5节给出的mnist_inference.py代码计算神经网络前向传播的结果。
    y = mnist_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)

    # 计算损失函数并定义反向传播过程。
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE, global_step, 60000 / BATCH_SIZE, LEARNING_RATE_DECAY)
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 定义每一轮迭代需要运行的操作。
    if is_chief:
        # 计算变量的滑动平均值。
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
        with tf.control_dependencies([variables_averages_op, train_op]):
            train_op = tf.no_op()
    return global_step, loss, train_op


def main(argv=None):
    # print('job_name=%s, task_id=%s' % (FLAGS.job_name, FLAGS.task_id))
    # 解析flags并通过tf.train.ClusterSpec配置TensorFlow集群。
    ps_hosts = FLAGS.ps_hosts.split(',')
    worker_hosts = FLAGS.worker_hosts.split(',')
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    # 通过tf.train.ClusterSpec以及当前任务创建tf.train.Server。
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_id)

    # 参数服务器只需要管理TensorFlow中的变量，不需要执行训练的过程。server.join()会
    # 一致停在这条语句上。
    if FLAGS.job_name == 'ps':
        with tf.device("/cpu:0"):
            server.join()

    # 定义计算服务器需要运行的操作。
    is_chief = (FLAGS.task_id == 0)
    mnist = input_data.read_data_sets(DATA_PATH, one_hot=True)

    device_setter = tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_id, cluster=cluster)
    with tf.device(device_setter):

        # 定义输入并得到每一轮迭代需要运行的操作。
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
        global_step, loss, train_op = build_model(x, y_, is_chief)

        # 定义用于保存模型的saver。
        saver = tf.train.Saver()
        # 定义日志输出操作。
        summary_op = tf.summary.merge_all()
        # 定义变量初始化操作。
        init_op = tf.global_variables_initializer()
        # 通过tf.train.Supervisor管理训练深度学习模型时的通用功能。
        sv = tf.train.Supervisor(
            is_chief=is_chief,        # 定义当前计算服务器是否为主计算服务器，只有主计算服务器会保存模型以及输出日志
            logdir=MODEL_SAVE_PATH,   # 指定保存模型和输出日志的地址
            init_op=init_op,          # 指定初始化操作
            summary_op=summary_op,    # 指定日志生成操作
            saver=saver,              # 指定用于保存模型的saver
            global_step=global_step,  # 指定当前迭代的轮数，这个会用于生成保存模型文件的文件名
            save_model_secs=60,       # 指定保存模型的时间间隔
            save_summaries_secs=60)   # 指定日志输出的时间间隔

        sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        # 通过tf.train.Supervisor生成会话。
        sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)

        step = 0
        start_time = time.time()

        # 执行迭代过程。
        while not sv.should_stop():
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, global_step_value = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if global_step_value >= TRAINING_STEPS: break

            # 每隔一段时间输出训练信息。
            if step > 0 and step % 100 == 0:
                duration = time.time() - start_time
                sec_per_batch = duration / global_step_value
                format_str = "After %d training steps (%d global steps), loss on training batch is %g.  (%.3f sec/batch)"
                print(format_str % (step, global_step_value, loss_value, sec_per_batch))
            step += 1
    sv.stop()


if __name__ == "__main__":
    tf.app.run()

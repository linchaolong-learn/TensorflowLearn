import tensorflow as tf
from tensorflow import ConfigProto

a = tf.constant('Hello, distribute tensorflow server1')

# 生成一个有两个任务的集群，一个任务跑在被内地2222端口，另外一个跑在本地2233端口
cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})

# 通过上面生成的集群配置生成Server，并通过job_name和task_index指定当前所启动的
# 任务。因为该任务是第一个任务，所以task_index为0
server = tf.train.Server(cluster, job_name="local", task_index=0)

# 通过server.target生成会话来使用TensorFlow集群中的资源。通过设置
# log_device_placement可以看到执行每一个操作的任务
sess = tf.Session(server.target, config=ConfigProto(log_device_placement=True, allow_soft_placement=True))
print(sess.run(a))

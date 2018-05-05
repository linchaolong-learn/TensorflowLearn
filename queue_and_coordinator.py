import numpy as np
import threading
import time
import tensorflow as tf


# 线程中运行的程序，这个程序每隔1秒判断是否需要停止并打印自己的ID
def MyLoop(coord, worker_id):
    # 使用 tf.Coordinator 类提供的协同工具判断当前线程是否需要停止
    while not coord.should_stop():
        # 随机停止所有的线程
        if np.random.rand() < 0.1:
            print("Stoping from id: %d\n" % worker_id)
            # 调用 coord.request_stop() 函数来通知其他线程停止
            coord.request_stop()
        else:
            # 打印当前线程的Id
            print("Working on id: %d\n" % worker_id)
        # 暂停1秒
        time.sleep(1)


# 声明一个 tf.train.Coordinator
coord = tf.train.Coordinator()
# 声明创建5个线程
# target 接收一个 function，用于线程中执行
# args 是该 function 的参数列表
threads = [threading.Thread(target=MyLoop, args=(coord, i)) for i in range(5)]

# 启动所有线程
for t in threads:
    t.start()

# 等待所有线程退出
coord.join(threads)

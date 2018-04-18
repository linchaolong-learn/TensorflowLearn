from tensorflow.examples.tutorials.mnist import input_data

# 载入MNIST数据集
mnist = input_data.read_data_sets('mnist_data')

print('train size:', mnist.train.num_examples)
print('validation size:', mnist.validation.num_examples)
print('test size:', mnist.test.num_examples)

# print('train.images[0]', mnist.train.images[0])
# print('train.labels[0]', mnist.train.labels[0])


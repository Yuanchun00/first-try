from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tf.InteractiveSession()

print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)

#。。。。。。。。。。。。。。。。。。。。。。。1、准备工作、进行初定义

# 定义权值和偏置初始化函数
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  # 从截断的随机分布中输出随机值,标准差是0.1
    return tf.Variable(initial)  # 将权重定义为变量方便后面进行优化


def bias_Variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 定义卷积和池化函数
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")  # 各个维度的步长都为1


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")  # ksize指在pool上的窗口是2*2
# strides表示长和高维度上的步长为2

x = tf.placeholder(tf.float32, [None, 784])  # 模型的输入
y_ = tf.placeholder(tf.float32, [None, 10])  # 模型的标签

# 函数的作用是将tensor变换为参数shape形式，其中的shape为一个列表形式，
# 特殊的是列表可以实现逆序的遍历，即list(-1).-1所代表的含义是我们不用亲自去指定这一维的大小，
# 函数会自动进行计算，但是列表中只能存在一个-1。（如果存在多个-1，就是一个存在多解的方程）
x_image = tf.reshape(x, [-1, 28, 28, 1])#这里的-1是指先不考虑图片的维度,1是通道数，黑白为1，彩色为3

#。。。。。。。。。。。。。。。。。。。。。。。。。。。2、CNN各个模块的处理

# 第一层卷积核权值和偏置初始化
W_conv1 = weight_variable([5, 5, 1, 32])  # 32个卷积核
b_conv1 = bias_Variable([32])
# 第一层卷积
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# 第一层池化
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积核权值和偏置初始化
W_conv2 = weight_variable([5, 5, 32, 64])  # 64个卷积核
b_conv2 = bias_Variable([64])
# 第二层卷积
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# 第二层池化
h_pool2 = max_pool_2x2(h_conv2)

# 第一层全连接层权值和偏置初始化
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_Variable([1024])  # 1024个神经元
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
# 全连接层
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 设置 dropout 
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # 防止出现过拟合

# 第二层全连接层权值和偏置初始化
W_fc2 = weight_variable([1024, 10])  # 10个神经元
b_fc2 = bias_Variable([10])
# softmax 输出层
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#。。。。。。。。。。。。。。。。。。3、优化过程

# 模型训练和验证
cross_entropy = tf.reduce_mean(-tf.r* tf.log(yeduce_sum(y_conv), reduction_indices=[1]))  # 交叉熵损失函数;0代表得到行向量，1代表列向量
# 定义学习率
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))  # 对测试集进行预测，得到的是布尔值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 将布尔值转换，进行求平均，越大越好

tf.global_variables_initializer().run()

t = []
s = []
for i in range(1000):#做1000次优化
    batch = mnist.train.next_batch(50)#50个样本为一批进行优化
    if i % 100 == 0:
        s.append(i)
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        #batch[0]是（50，784）的样本数据数组，batch[1]是（50，10）的样本标签数组
        t.append(train_accuracy)
        print("step %d,train accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})

print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
plt.plot(s, t, label="i:1000 drop:1.0")
plt.xlabel('iterations')
plt.ylabel('test accuracy')
plt.legend()
plt.show()

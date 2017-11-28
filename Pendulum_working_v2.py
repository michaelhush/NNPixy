import scipy.io as spio
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

mat = spio.loadmat('data_oh_v3.mat', squeeze_me=True)
imported_y_ = mat['labels']
imported_x = mat['data']

imported_x = list(zip(*imported_x))  # transposes rows and columns
imported_y_ = list(zip(*imported_y_))

x_train, x_test, y_train, y_test = train_test_split(imported_x, imported_y_, test_size=0.25,
                                                    random_state=2)  # this shuffles the data as well

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

#x_train = np.concatenate((x_train, x_train))# artificially increase the training dataset by 2 times to see outcome
#y_train = np.concatenate((y_train, y_train))

x_ar_size = x_train.shape[1]# gets the size of the dataset per iteration
print(x_train.shape[0])
num_train_iterations = round(x_train.shape[0], -1)# takes the number of training iterations and rounds to nearest 10 for the training step size

# from the MNIST tutorial
x = tf.placeholder(tf.float32, [None, x_ar_size])
W = tf.Variable(tf.zeros([x_ar_size, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# functions to take batches from the training data
def x_train_batch(int_in, num_train_iterations):
    x_batch = x_train[int(int_in * num_train_iterations/10):int(int_in * num_train_iterations/10 + num_train_iterations/10 - 1), :]
    return x_batch

def y_train_batch(int_in, num_train_iterations):
    y_batch = y_train[int(int_in * num_train_iterations/10):int(int_in * num_train_iterations/10 + num_train_iterations/10 - 1), :]
    return y_batch

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for i in range(10):#weird things happen when you change this number
    yb = y_train_batch(i, num_train_iterations)
    xb = x_train_batch(i, num_train_iterations)
    sess.run(train_step, feed_dict={x: xb, y_: yb})

print(sess.run(accuracy, feed_dict={x: x_test, y_: y_test}))
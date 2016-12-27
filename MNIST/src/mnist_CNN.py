from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import csv as csv
from datetime import datetime
startTime = datetime.now()

NUM_DIGITS = 10
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def LoadMNISTData(trainDataPath,testDataPath):
    print("Read MNIST Training Data")
    csv_file_object = csv.reader(open(trainDataPath, 'rb'))
    header = csv_file_object.next()
    data=[]
    labels = []
    for row in csv_file_object:
        data.append(row[0:])
        # data.append(row[1:])
    train_data = np.array(data)
    # train_labels = np.array(labels)
    print("Read MNIST Testing Data")
    csv_file_object = csv.reader(open(testDataPath, 'rb'))
    header = csv_file_object.next()
    test_data=[]
    for row in csv_file_object:
        test_data.append(row[0:])
    test_data = np.array(test_data)

    return train_data,test_data


def savePredicationOutput(output,predicationPath):
    predictions_file = open(predicationPath, "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["ImageId", "Label"])
    for idx,label in enumerate(output):
        open_file_object.writerow([idx+1, label])
    predictions_file.close()

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

print("Loading MNIST Data")
train_data,test_data = LoadMNISTData('./data/train.csv', './data/test.csv')

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
predication_output = tf.argmax(y_conv,1)
correct_prediction = tf.equal(predication_output, tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

for i in range(20000):
    np.random.shuffle(train_data)
    data = train_data[:50, 1:]
    labels = train_data[:50, 0]
    # batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:data, y_: np.eye(NUM_DIGITS)[labels.astype(int)], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: data, y_: np.eye(NUM_DIGITS)[labels.astype(int)], keep_prob: 0.5})

# save_path = saver.save(sess, "/home/ahmed/Work/Tutorials/Kaggle/MNIST/model/kaggleModel/cnn_model.ckpt")
# print("Model saved in file: %s" % save_path)

# print("test accuracy %g"%accuracy.eval(feed_dict={
#     x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
output = []
for idx in range(0, test_data.shape[0],50):
    batch = test_data[idx:idx+50, :]
    batch_output = sess.run(predication_output, feed_dict={x: batch, y_: np.zeros((batch.shape[0], NUM_DIGITS)), keep_prob: 1.0})
    for label in batch_output:
        output.append(label)

print("Saving Predication Output")
savePredicationOutput(output, "./submission/cnnOutput.csv")
print("Done in " + str(datetime.now() - startTime))
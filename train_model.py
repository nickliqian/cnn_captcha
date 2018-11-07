# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image
import random
import os

# 设置以下环境变量可开启CPU识别
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def gen_captcha_text_image(img_name):
    """
    返回一个验证码的array形式和对应的字符串标签
    :return:tuple (str, numpy.array)
    """
    # 标签
    label = img_name.split("_")[0]
    # 文件
    img_file = os.path.join(img_path, img_name)
    captcha_image = Image.open(img_file)
    captcha_array = np.array(captcha_image)  # 向量化
    return label, captcha_array


def convert2gray(img):
    """
    图片转为灰度图，如果是3通道图则计算，单通道图则直接返回
    :param img:
    :return:
    """
    if len(img.shape) > 2:
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img


def text2vec(text):
    """
    转标签为oneHot编码
    :param text: str
    :return: numpy.array
    """
    text_len = len(text)
    if text_len > max_captcha:
        raise ValueError('验证码最长{}个字符'.format(max_captcha))

    vector = np.zeros(max_captcha * char_set_len)

    for i, ch in enumerate(text):
        idx = i * char_set_len + char_set.index(ch)
        vector[idx] = 1
    return vector


def get_batch(n, size=128):
    batch_x = np.zeros([size, image_height * image_width])  # 初始化
    batch_y = np.zeros([size, max_captcha * char_set_len])  # 初始化

    max_batch = int(len(img_list) / size)
    if n > max_batch - 1:
        n = n % max_batch
    s = n * size
    e = (n + 1) * size
    this_batch = img_list[s:e]
    print("{}:{}".format(s, e))

    for i, img_name in enumerate(this_batch):
        label, image_array = gen_captcha_text_image(img_name)
        image_array = convert2gray(image_array)  # 灰度化图片
        batch_x[i, :] = image_array.flatten() / 255  # flatten 转为一维
        batch_y[i, :] = text2vec(label)  # 生成 oneHot
    return batch_x, batch_y


def model():
    x = tf.reshape(X, shape=[-1, image_height, image_width, 1])
    print(">>> input x: {}".format(x))

    # 卷积层1
    wc1 = tf.get_variable(name='wc1', shape=[3, 3, 1, 32], dtype=tf.float32,
                          initializer=tf.contrib.layers.xavier_initializer())
    # wc1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
    bc1 = tf.Variable(b_alpha * tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, wc1, strides=[1, 1, 1, 1], padding='SAME'), bc1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)

    # 卷积层2
    wc2 = tf.get_variable(name='wc2', shape=[3, 3, 32, 64], dtype=tf.float32,
                          initializer=tf.contrib.layers.xavier_initializer())
    # wc2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
    bc2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, wc2, strides=[1, 1, 1, 1], padding='SAME'), bc2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    # 卷积层3
    wc3 = tf.get_variable(name='wc3', shape=[3, 3, 64, 128], dtype=tf.float32,
                          initializer=tf.contrib.layers.xavier_initializer())
    # wc3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 128]))
    bc3 = tf.Variable(b_alpha * tf.random_normal([128]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, wc3, strides=[1, 1, 1, 1], padding='SAME'), bc3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)
    print(">>> convolution 3: ", conv3.shape)
    next_shape = conv3.shape[1] * conv3.shape[2] * conv3.shape[3]

    # 全连接层1
    wd1 = tf.get_variable(name='wd1', shape=[next_shape, 1024], dtype=tf.float32,
                          initializer=tf.contrib.layers.xavier_initializer())
    # wd1 = tf.Variable(w_alpha * tf.random_normal([7*20*128,1024]))
    bd1 = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.reshape(conv3, [-1, wd1.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, wd1), bd1))
    dense = tf.nn.dropout(dense, keep_prob)

    # 全连接层2
    wout = tf.get_variable('name', shape=[1024, max_captcha * char_set_len], dtype=tf.float32,
                           initializer=tf.contrib.layers.xavier_initializer())
    # wout = tf.Variable(w_alpha * tf.random_normal([1024, max_captcha * char_set_len]))
    bout = tf.Variable(b_alpha * tf.random_normal([max_captcha * char_set_len]))
    y_predict = tf.add(tf.matmul(dense, wout), bout)
    return y_predict


def train_cnn():
    y_predict = model()
    print(">>> input batch predict shape: {}".format(y_predict.shape))
    print(">>> End model test")
    # 计算概率 损失
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_predict, labels=Y))
    # 梯度下降
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
    # 计算准确率
    predict = tf.reshape(y_predict, [-1, max_captcha, char_set_len])  # 预测结果
    max_idx_p = tf.argmax(predict, 2)  # 预测结果
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, max_captcha, char_set_len]), 2)  # 标签
    # 计算准确率
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    # 模型保存对象
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        # 恢复模型
        if os.path.exists("./model/"):
            saver.restore(sess, "./model/")
        else:
            pass
        step = 1
        for i in range(3000):
            batch_x, batch_y = get_batch(i, size=128)
            _, cost_ = sess.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})
            if step % 10 == 0:
                batch_x_test, batch_y_test = get_batch(i, size=100)
                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                print("第{}次训练 >>> 准确率为 {} >>> loss {}".format(step, acc, cost_))
                # if acc > 0.99:
                #     saver.save(sess, "./model/")
                #     break
            if i % 500 == 0:
                saver.save(sess, "./model/")
            step += 1
        saver.save(sess, "./model/")


def crack_captcha(captcha_image):
    y_predict = model()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "./model/")
        predict = tf.argmax(tf.reshape(y_predict, [-1, max_captcha, char_set_len]), 2)
        text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1.})
        text = text_list[0].tolist()
        return text


def main():
    global image_height
    global image_width
    global max_captcha
    global char_set
    global char_set_len
    global X
    global Y
    global keep_prob
    # 图片宽高和字符长度基本信息
    print("图片总数：{}".format(len(img_list)))
    exp_text, image = gen_captcha_text_image(img_list[0])
    image_height, image_width, channel = image.shape
    max_captcha = len(exp_text)  # 验证码位数

    # 验证码可能的字符和总类别
    char_set = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f',
                'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    char_set_len = len(char_set)

    # 相关信息打印
    print("-->图片尺寸: {}".format(image.shape))
    print("-->验证码长度: {}".format(max_captcha))
    print("-->验证码共{}类 {}".format(char_set_len, char_set))

    # tf初始化占位符
    X = tf.placeholder(tf.float32, [None, image_height * image_width])  # 特征向量
    Y = tf.placeholder(tf.float32, [None, max_captcha * char_set_len])  # 标签
    keep_prob = tf.placeholder(tf.float32)  # dropout值

    # test model input and output
    print(">>> Start model test")
    batch_x, batch_y = get_batch(0, size=100)
    print(">>> input batch images shape: {}".format(batch_x.shape))
    print(">>> input batch labels shape: {}".format(batch_y.shape))

    train = 1
    if train == 1:
        train_cnn()
    elif train == 0:
        f = plt.figure()
        ax = f.add_subplot(111)
        ax.text(0.1, 0.9, "origin:" + exp_text, ha='center', va='center', transform=ax.transAxes)
        plt.imshow(image)
        # 预测图片
        image = convert2gray(image)
        image = image.flatten() / 255
        predict_text = crack_captcha(image)
        print("正确: {}  预测: {}".format(exp_text, predict_text))
        # 显示图片和预测结果
        p_text = ""
        for p in predict_text:
            p_text += str(char_set[p])
        print(p_text)
        plt.text(20, 1, 'predict:{}'.format(p_text))
        plt.show()


if __name__ == '__main__':
    # 全局变量
    image_height = 0
    image_width = 0
    max_captcha = 0
    char_set = []
    char_set_len = 0
    X = None
    Y = None
    keep_prob = None
    w_alpha = 0.01
    b_alpha = 0.1
    img_path = "./sample/train"
    img_list = os.listdir(img_path)
    random.seed(time.time())
    random.shuffle(img_list)
    main()

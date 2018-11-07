# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import time
from PIL import Image
import random
import os


def gen_captcha_text_image():
    """
    返回一个验证码的array形式和对应的字符串标签
    :return:tuple (str, numpy.array)
    """
    img_name = random.choice(img_list)
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
    next_shape = conv3.shape[1]*conv3.shape[2]*conv3.shape[3]

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


def test_batch():
    y_predict = model()
    total = 1000
    right = 0

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "./model/")
        s = time.time()
        for i in range(total):
            # test_text, test_image = gen_special_num_image(i)
            test_text, test_image = gen_captcha_text_image()  # 随机
            test_image = convert2gray(test_image)
            test_image = test_image.flatten() / 255

            predict = tf.argmax(tf.reshape(y_predict, [-1, max_captcha, char_set_len]), 2)
            text_list = sess.run(predict, feed_dict={X: [test_image], keep_prob: 1.})
            preidct_text = text_list[0].tolist()
            p_text = ""
            for p in preidct_text:
                p_text += str(char_set[p])
            print("origin: {} predict: {}".format(test_text, p_text))
            if test_text == p_text:
                right += 1
            else:
                pass
        e = time.time()
    rate = str(right/total) + "%"
    print("测试结果： {}/{}".format(right, total))
    print("{}个样本识别耗时{}秒，准确率{}".format(total, e-s, rate))


# char_set = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
# char_set = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
def main():
    global image_height
    global image_width
    global max_captcha
    global char_set
    global char_set_len
    global X
    global Y
    global keep_prob
    global test_image_label_list

    # 图片宽高和字符长度基本信息
    exp_text, image = gen_captcha_text_image()
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
    print("-->使用测试集为 {}".format(img_path))

    # tf初始化占位符
    X = tf.placeholder(tf.float32, [None, image_height * image_width])  # 特征向量
    Y = tf.placeholder(tf.float32, [None, max_captcha * char_set_len])  # 标签
    keep_prob = tf.placeholder(tf.float32)  # dropout值
    test_batch()


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
    img_path = "./sample/test"
    img_list = os.listdir(img_path)
    random.seed(time.time())
    random.shuffle(img_list)
    main()

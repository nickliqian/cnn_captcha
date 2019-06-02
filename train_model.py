# -*- coding: utf-8 -*-
import json

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image
import random
import os
from cnnlib.network import CNN


class TrainError(Exception):
    pass


class TrainModel(CNN):
    def __init__(self, train_img_path, verify_img_path, char_set, model_save_dir, cycle_stop, acc_stop, cycle_save, image_suffix, verify=False):
        # 训练相关参数
        self.cycle_stop = cycle_stop
        self.acc_stop = acc_stop
        self.cycle_save = cycle_save

        self.image_suffix = image_suffix
        char_set = [str(i) for i in char_set]

        # 打乱文件顺序+校验图片格式
        self.train_img_path = train_img_path
        self.train_images_list = os.listdir(train_img_path)
        # 校验格式
        if verify:
            self.confirm_image_suffix()
        # 打乱文件顺序
        random.seed(time.time())
        random.shuffle(self.train_images_list)

        # 验证集文件
        self.verify_img_path = verify_img_path
        self.verify_images_list = os.listdir(verify_img_path)

        # 获得图片宽高和字符长度基本信息
        label, captcha_array = self.gen_captcha_text_image(train_img_path, self.train_images_list[0])

        captcha_shape = captcha_array.shape
        captcha_shape_len = len(captcha_shape)
        if captcha_shape_len == 3:
            image_height, image_width, channel = captcha_shape
            self.channel = channel
        elif captcha_shape_len == 2:
            image_height, image_width = captcha_shape
        else:
            raise TrainError("图片转换为矩阵时出错，请检查图片格式")

        # 初始化变量
        super(TrainModel, self).__init__(image_height, image_width, len(label), char_set, model_save_dir)

        # 相关信息打印
        print("-->图片尺寸: {} X {}".format(image_height, image_width))
        print("-->验证码长度: {}".format(self.max_captcha))
        print("-->验证码共{}类 {}".format(self.char_set_len, char_set))
        print("-->使用测试集为 {}".format(train_img_path))
        print("-->使验证集为 {}".format(verify_img_path))

        # test model input and output
        print(">>> Start model test")
        batch_x, batch_y = self.get_batch(0, size=100)
        print(">>> input batch images shape: {}".format(batch_x.shape))
        print(">>> input batch labels shape: {}".format(batch_y.shape))

    @staticmethod
    def gen_captcha_text_image(img_path, img_name):
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

    def get_batch(self, n, size=128):
        batch_x = np.zeros([size, self.image_height * self.image_width])  # 初始化
        batch_y = np.zeros([size, self.max_captcha * self.char_set_len])  # 初始化

        max_batch = int(len(self.train_images_list) / size)
        # print(max_batch)
        if max_batch - 1 < 0:
            raise TrainError("训练集图片数量需要大于每批次训练的图片数量")
        if n > max_batch - 1:
            n = n % max_batch
        s = n * size
        e = (n + 1) * size
        this_batch = self.train_images_list[s:e]
        # print("{}:{}".format(s, e))

        for i, img_name in enumerate(this_batch):
            label, image_array = self.gen_captcha_text_image(self.train_img_path, img_name)
            image_array = self.convert2gray(image_array)  # 灰度化图片
            batch_x[i, :] = image_array.flatten() / 255  # flatten 转为一维
            batch_y[i, :] = self.text2vec(label)  # 生成 oneHot
        return batch_x, batch_y

    def get_verify_batch(self, size=100):
        batch_x = np.zeros([size, self.image_height * self.image_width])  # 初始化
        batch_y = np.zeros([size, self.max_captcha * self.char_set_len])  # 初始化

        verify_images = []
        for i in range(size):
            verify_images.append(random.choice(self.verify_images_list))

        for i, img_name in enumerate(verify_images):
            label, image_array = self.gen_captcha_text_image(self.verify_img_path, img_name)
            image_array = self.convert2gray(image_array)  # 灰度化图片
            batch_x[i, :] = image_array.flatten() / 255  # flatten 转为一维
            batch_y[i, :] = self.text2vec(label)  # 生成 oneHot
        return batch_x, batch_y

    def confirm_image_suffix(self):
        # 在训练前校验所有文件格式
        print("开始校验所有图片后缀")
        for index, img_name in enumerate(self.train_images_list):
            print("{} image pass".format(index), end='\r')
            if not img_name.endswith(self.image_suffix):
                raise TrainError('confirm images suffix：you request [.{}] file but get file [{}]'
                                 .format(self.image_suffix, img_name))
        print("所有图片格式校验通过")

    def train_cnn(self):
        y_predict = self.model()
        print(">>> input batch predict shape: {}".format(y_predict.shape))
        print(">>> End model test")
        # 计算概率 损失
        with tf.name_scope('cost'):
            cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_predict, labels=self.Y))
        # 梯度下降
        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
        # 计算准确率
        predict = tf.reshape(y_predict, [-1, self.max_captcha, self.char_set_len])  # 预测结果
        max_idx_p = tf.argmax(predict, 2)  # 预测结果
        max_idx_l = tf.argmax(tf.reshape(self.Y, [-1, self.max_captcha, self.char_set_len]), 2)  # 标签
        # 计算准确率
        correct_pred = tf.equal(max_idx_p, max_idx_l)
        with tf.name_scope('char_acc'):
            accuracy_char_count = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        with tf.name_scope('image_acc'):
            accuracy_image_count = tf.reduce_mean(tf.reduce_min(tf.cast(correct_pred, tf.float32), axis=1))
        # 模型保存对象
        saver = tf.train.Saver()
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            # 恢复模型
            if os.path.exists(self.model_save_dir):
                try:
                    saver.restore(sess, self.model_save_dir)
                # 判断捕获model文件夹中没有模型文件的错误
                except ValueError:
                    print("model文件夹为空，将创建新模型")
            else:
                pass
            # 写入日志
            tf.summary.FileWriter("logs/", sess.graph)

            step = 1
            for i in range(self.cycle_stop):
                batch_x, batch_y = self.get_batch(i, size=128)
                # 梯度下降训练
                _, cost_ = sess.run([optimizer, cost],
                                    feed_dict={self.X: batch_x, self.Y: batch_y, self.keep_prob: 0.75})
                if step % 10 == 0:
                    # 基于训练集的测试
                    batch_x_test, batch_y_test = self.get_batch(i, size=100)
                    acc_char = sess.run(accuracy_char_count, feed_dict={self.X: batch_x_test, self.Y: batch_y_test, self.keep_prob: 1.})
                    acc_image = sess.run(accuracy_image_count, feed_dict={self.X: batch_x_test, self.Y: batch_y_test, self.keep_prob: 1.})
                    print("第{}次训练 >>> ".format(step))
                    print("[训练集] 字符准确率为 {:.5f} 图片准确率为 {:.5f} >>> loss {:.10f}".format(acc_char, acc_image, cost_))

                    # with open("loss_train.csv", "a+") as f:
                    #     f.write("{},{},{},{}\n".format(step, acc_char, acc_image, cost_))

                    # 基于验证集的测试
                    batch_x_verify, batch_y_verify = self.get_verify_batch(size=100)
                    acc_char = sess.run(accuracy_char_count, feed_dict={self.X: batch_x_verify, self.Y: batch_y_verify, self.keep_prob: 1.})
                    acc_image = sess.run(accuracy_image_count, feed_dict={self.X: batch_x_verify, self.Y: batch_y_verify, self.keep_prob: 1.})
                    print("[验证集] 字符准确率为 {:.5f} 图片准确率为 {:.5f} >>> loss {:.10f}".format(acc_char, acc_image, cost_))

                    # with open("loss_test.csv", "a+") as f:
                    #     f.write("{}, {},{},{}\n".format(step, acc_char, acc_image, cost_))

                    # 准确率达到99%后保存并停止
                    if acc_image > self.acc_stop:
                        saver.save(sess, self.model_save_dir)
                        print("验证集准确率达到99%，保存模型成功")
                        break
                # 每训练500轮就保存一次
                if i % self.cycle_save == 0:
                    saver.save(sess, self.model_save_dir)
                    print("定时保存模型成功")
                step += 1
            saver.save(sess, self.model_save_dir)

    def recognize_captcha(self):
        label, captcha_array = self.gen_captcha_text_image(self.train_img_path, random.choice(self.train_images_list))

        f = plt.figure()
        ax = f.add_subplot(111)
        ax.text(0.1, 0.9, "origin:" + label, ha='center', va='center', transform=ax.transAxes)
        plt.imshow(captcha_array)
        # 预测图片
        image = self.convert2gray(captcha_array)
        image = image.flatten() / 255

        y_predict = self.model()

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.model_save_dir)
            predict = tf.argmax(tf.reshape(y_predict, [-1, self.max_captcha, self.char_set_len]), 2)
            text_list = sess.run(predict, feed_dict={self.X: [image], self.keep_prob: 1.})
            predict_text = text_list[0].tolist()

        print("正确: {}  预测: {}".format(label, predict_text))
        # 显示图片和预测结果
        p_text = ""
        for p in predict_text:
            p_text += str(self.char_set[p])
        print(p_text)
        plt.text(20, 1, 'predict:{}'.format(p_text))
        plt.show()


def main():
    with open("conf/sample_config.json", "r") as f:
        sample_conf = json.load(f)

    train_image_dir = sample_conf["train_image_dir"]
    verify_image_dir = sample_conf["test_image_dir"]
    model_save_dir = sample_conf["model_save_dir"]
    cycle_stop = sample_conf["cycle_stop"]
    acc_stop = sample_conf["acc_stop"]
    cycle_save = sample_conf["cycle_save"]
    enable_gpu = sample_conf["enable_gpu"]
    image_suffix = sample_conf['image_suffix']
    use_labels_json_file = sample_conf['use_labels_json_file']

    if use_labels_json_file:
        with open("tools/labels.json", "r") as f:
            char_set = f.read().strip()
    else:
        char_set = sample_conf["char_set"]

    if not enable_gpu:
        # 设置以下环境变量可开启CPU识别
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    tm = TrainModel(train_image_dir, verify_image_dir, char_set, model_save_dir, cycle_stop, acc_stop, cycle_save, image_suffix, verify=False)
    tm.train_cnn()  # 开始训练模型
    # tm.recognize_captcha()  # 识别图片示例


if __name__ == '__main__':
    main()

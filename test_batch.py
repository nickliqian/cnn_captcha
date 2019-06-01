# -*- coding: utf-8 -*-
import json

import tensorflow as tf
import numpy as np
import time
from PIL import Image
import random
import os
from cnnlib.network import CNN


class TestError(Exception):
    pass


class TestBatch(CNN):
    def __init__(self, img_path, char_set, model_save_dir, total):
        # 模型路径
        self.model_save_dir = model_save_dir
        # 打乱文件顺序
        self.img_path = img_path
        self.img_list = os.listdir(img_path)
        random.seed(time.time())
        random.shuffle(self.img_list)

        # 获得图片宽高和字符长度基本信息
        label, captcha_array = self.gen_captcha_text_image()

        captcha_shape = captcha_array.shape
        captcha_shape_len = len(captcha_shape)
        if captcha_shape_len == 3:
            image_height, image_width, channel = captcha_shape
            self.channel = channel
        elif captcha_shape_len == 2:
            image_height, image_width = captcha_shape
        else:
            raise TestError("图片转换为矩阵时出错，请检查图片格式")

        # 初始化变量
        super(TestBatch, self).__init__(image_height, image_width, len(label), char_set, model_save_dir)
        self.total = total

        # 相关信息打印
        print("-->图片尺寸: {} X {}".format(image_height, image_width))
        print("-->验证码长度: {}".format(self.max_captcha))
        print("-->验证码共{}类 {}".format(self.char_set_len, char_set))
        print("-->使用测试集为 {}".format(img_path))

    def gen_captcha_text_image(self):
        """
        返回一个验证码的array形式和对应的字符串标签
        :return:tuple (str, numpy.array)
        """
        img_name = random.choice(self.img_list)
        # 标签
        label = img_name.split("_")[0]
        # 文件
        img_file = os.path.join(self.img_path, img_name)
        captcha_image = Image.open(img_file)
        captcha_array = np.array(captcha_image)  # 向量化

        return label, captcha_array

    def test_batch(self):
        y_predict = self.model()
        total = self.total
        right = 0

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.model_save_dir)
            s = time.time()
            for i in range(total):
                # test_text, test_image = gen_special_num_image(i)
                test_text, test_image = self.gen_captcha_text_image()  # 随机
                test_image = self.convert2gray(test_image)
                test_image = test_image.flatten() / 255

                predict = tf.argmax(tf.reshape(y_predict, [-1, self.max_captcha, self.char_set_len]), 2)
                text_list = sess.run(predict, feed_dict={self.X: [test_image], self.keep_prob: 1.})
                predict_text = text_list[0].tolist()
                p_text = ""
                for p in predict_text:
                    p_text += str(self.char_set[p])
                print("origin: {} predict: {}".format(test_text, p_text))
                if test_text == p_text:
                    right += 1
                else:
                    pass
            e = time.time()
        rate = str(right/total * 100) + "%"
        print("测试结果： {}/{}".format(right, total))
        print("{}个样本识别耗时{}秒，准确率{}".format(total, e-s, rate))


def main():
    with open("conf/sample_config.json", "r") as f:
        sample_conf = json.load(f)

    test_image_dir = sample_conf["test_image_dir"]
    model_save_dir = sample_conf["model_save_dir"]

    use_labels_json_file = sample_conf['use_labels_json_file']

    if use_labels_json_file:
        with open("tools/labels.json", "r") as f:
            char_set = f.read().strip()
    else:
        char_set = sample_conf["char_set"]

    total = 100
    tb = TestBatch(test_image_dir, char_set, model_save_dir, total)
    tb.test_batch()


if __name__ == '__main__':
    main()

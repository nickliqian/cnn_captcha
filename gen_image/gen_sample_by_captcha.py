# -*- coding: UTF-8 -*-
"""
使用captcha lib生成验证码（前提：pip install captcha）
"""
from captcha.image import ImageCaptcha
import os
import random
import time


def gen_special_img(text, file_path):
    # 生成img文件
    generator = ImageCaptcha(width=width, height=height)  # 指定大小
    img = generator.generate_image(text)  # 生成图片
    img.save(file_path)  # 保存图片


if __name__ == '__main__':
    # 配置参数
    root_dir = "../sample/python_captcha/"  # 图片储存路径
    image_suffix = "png"  # 图片储存后缀
    characters = "0123456789"  # 图片上显示的字符集
    # characters = "0123456789abcdefghijklmnopqrstuvwxyz"
    count = 100  # 生成多少张样本
    char_count = 4  # 图片上的字符数量

    # 设置图片高度和宽度
    width = 100
    height = 60

    # 判断文件夹是否存在
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)

    for i in range(count):
        text = ""
        for j in range(char_count):
            text += random.choice(characters)
        timec = str(time.time()).replace(".", "")
        p = os.path.join(root_dir, "{}_{}.{}".format(text, timec, image_suffix))
        gen_special_img(text, p)


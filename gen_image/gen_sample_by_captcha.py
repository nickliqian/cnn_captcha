# -*- coding: UTF-8 -*-
"""
使用captcha lib生成验证码（前提：pip install captcha）
"""
from captcha.image import ImageCaptcha
import os
import random
import time


def gen_special_img(text, file_path):
    # 设置图片大小和宽度
    width = 100
    height = 60
    # 生成img文件
    generator = ImageCaptcha(width=width, height=height)  # 指定大小
    img = generator.generate_image(text)  # 生成图片
    img.save(file_path)  # 保存图片


if __name__ == '__main__':
    root_dir = "../sample/python_captcha/"
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    characters = "0123456789"
    # characters = "0123456789abcdefghijklmnopqrstuvwxyz"
    for i in range(100):
        text = ""
        for j in range(4):
            text += random.choice(characters)
        timec = str(time.time()).replace(".", "")
        p = os.path.join(root_dir, "{}_{}.png".format(text, timec))
        gen_special_img(text, p)


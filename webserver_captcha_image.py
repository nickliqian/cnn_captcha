# -*- coding: UTF-8 -*-
"""
    验证码图片接口，访问`/captcha/1`获得图片
"""
from captcha.image import ImageCaptcha
import os
import random
from flask import Flask, request, jsonify, Response, make_response
import json
import io


# Flask对象
app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))


with open("conf/captcha_config.json", "r") as f:
    config = json.load(f)
# 配置参数
root_dir = config["root_dir"]  # 图片储存路径
image_suffix = config["image_suffix"]  # 图片储存后缀
characters = config["characters"]  # 图片上显示的字符集 # characters = "0123456789abcdefghijklmnopqrstuvwxyz"
count = config["count"]  # 生成多少张样本
char_count = config["char_count"]  # 图片上的字符数量

# 设置图片高度和宽度
width = config["width"]
height = config["height"]


def response_headers(content):
    resp = Response(content)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


def gen_special_img():
    # 随机文字
    text = ""
    for j in range(char_count):
        text += random.choice(characters)
    print(text)
    # 生成img文件
    generator = ImageCaptcha(width=width, height=height)  # 指定大小
    img = generator.generate_image(text)  # 生成图片
    imgByteArr = io.BytesIO()
    img.save(imgByteArr, format='PNG')
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr


@app.route('/captcha/', methods=['GET'])
def show_photo():
    if request.method == 'GET':
        image_data = gen_special_img()
        response = make_response(image_data)
        response.headers['Content-Type'] = 'image/png'
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response
    else:
        pass


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=6100,
        debug=True
    )

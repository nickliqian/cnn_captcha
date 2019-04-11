#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
人工在线验证脚本
"""
import requests
from io import BytesIO
import time
import matplotlib.pyplot as plt
import json
import numpy as np
from PIL import Image
import os


def correction(fail_path, pass_path, correction_times, remote_url):
    headers = {
        'user-agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.146 Safari/537.36",
    }

    fail_count = 0
    for index in range(correction_times):
        # 请求
        while True:
            try:
                response = requests.request("GET", remote_url, headers=headers, timeout=10)
                break
            except Exception as e:
                print(e)

        # 识别
        s = time.time()
        url = "http://127.0.0.1:6000/b"
        files = {'image_file': ('captcha.jpg', BytesIO(response.content), 'application')}
        r = requests.post(url=url, files=files)
        e = time.time()
        print(index, int((e-s)*1000), "ms")
        print(r.text)
        time.sleep(2)

        # 识别结果
        predict_text = json.loads(r.text)["value"]
        f = plt.figure()
        ax = f.add_subplot(111)
        ax.text(0.1, 0.9, "备注", ha='center', va='center', transform=ax.transAxes)

        # 图片字节流转为image array
        img = BytesIO(response.content)
        img = Image.open(img, mode="r")
        captcha_array = np.array(img)
        plt.imshow(captcha_array)

        # 预测图片
        print("预测: {}\n".format(predict_text))

        # 显示图片和预测结果
        plt.text(20, 2, 'predict:{}'.format(predict_text))
        plt.show()

        q = input("index:<{}> 正确按enter，错误输入真实值后会保存:".format(index))
        img_name = "{}_{}".format(q, str(time.time()).replace(".", ""))
        if q:
            path = os.path.join(fail_path, img_name)
            with open(path, "wb") as f:
                f.write(response.content)
            fail_count += 1
        else:
            path = os.path.join(pass_path, img_name)
            with open(path, "wb") as f:
                f.write(response.content)

        print("==============")

    rate = (correction_times - fail_count)/correction_times
    print("Pass Rate: {}".format(rate))


def main():
    fail_path = "./sample/fail_sample/"
    pass_path = "./sample/pass_sample/"
    correction_times = 10
    remote_url = "https://www.xxxxxxx.com/getImg"

    correction(fail_path, pass_path, correction_times, remote_url)


if __name__ == '__main__':
    main()




#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
使用自建的接口识别来自网络的验证码
需要配置参数：
    remote_url = "https://www.xxxxxxx.com/getImg"  验证码链接地址
    rec_times = 1  识别的次数
"""
import datetime
import requests
from io import BytesIO
import time
import json
import os


def recognize_captcha(index, test_path, save_path, image_suffix):
    image_file_name = 'captcha.{}'.format(image_suffix)

    with open(test_path, "rb") as f:
        content = f.read()

    # 识别
    s = time.time()
    url = "http://127.0.0.1:6000/b"
    files = {'image_file': (image_file_name, BytesIO(content), 'application')}
    r = requests.post(url=url, files=files)
    e = time.time()

    # 测试参数
    result_dict = json.loads(r.text)["value"]  # 响应
    predict_text = result_dict["value"]  # 识别结果
    whole_time_for_work = int((e - s) * 1000)
    speed_time_by_rec = result_dict["speed_time(ms)"]  # 模型识别耗时
    request_time_by_rec = whole_time_for_work - speed_time_by_rec  # 请求耗时
    now_time = datetime.datetime.now().strftime('%Y-%m-%d@%H:%M:%S')  # 当前时间

    # 记录日志
    log = "{},{},{},{},{},{}\n"\
        .format(index, predict_text, now_time, whole_time_for_work, speed_time_by_rec, request_time_by_rec)
    with open("./test.csv", "a+") as f:
        f.write(log)

    # 输出结果到控制台
    print("次数：{},结果：{},时刻：{},总耗时：{}ms,识别：{}ms,请求：{}ms"
          .format(index, predict_text, now_time, whole_time_for_work, speed_time_by_rec, request_time_by_rec))

    # 保存文件
    # img_name = "{}_{}.{}".format(predict_text, str(time.time()).replace(".", ""), image_suffix)
    # path = os.path.join(save_path, img_name)
    # with open(path, "wb") as f:
    #     f.write(content)


def main():
    with open("conf/sample_config.json", "r") as f:
        sample_conf = json.load(f)

    # 配置相关参数
    test_file = "sample/test/0001_15430304076164024.png"  # 测试识别的图片路径
    save_path = sample_conf["local_image_dir"]  # 保存的地址
    image_suffix = sample_conf["image_suffix"]  # 文件后缀
    for i in range(20000):
        recognize_captcha(i, test_file, save_path, image_suffix)


if __name__ == '__main__':
    main()
    


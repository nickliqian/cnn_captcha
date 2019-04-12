#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
统计样本的标签，并写入文件labels.json
"""
import os
import json


image_dir = "../sample/origin"
image_list = os.listdir(image_dir)

labels = set()
for img in image_list:
    split_result = img.split("_")
    if len(split_result) == 2:
        label, name = split_result
        if label:
            for word in label:
                labels.add(word)
    else:
        pass

print("共有标签{}种".format(len(labels)))

with open("./labels.json", "w") as f:
    f.write(json.dumps("".join(list(labels)), ensure_ascii=False))

print("将标签列表写入文件labels.json成功")

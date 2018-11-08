"""
验证图片尺寸和分离测试集（5%）和训练集（95%）
"""
from PIL import Image
import random
import os
import shutil
from sample import sample_conf


def verify(origin_dir, real_width, real_height):
    """
    校验图片大小
    :return:
    """
    print("开始校验原始图片集")
    # 图片真实尺寸
    real_size = (real_width, real_height)
    # 图片名称列表和数量
    img_list = os.listdir(origin_dir)
    total_count = len(img_list)

    # 无效图片列表
    bad_img = []
    print("Total image count: {}".format(len(img_list)))

    # 遍历所有图片进行验证
    for index, img_name in enumerate(img_list):
        file_path = os.path.join(origin_dir, img_name)
        img = Image.open(file_path)

        if real_size == img.size:
            print("{} pass".format(index), end='\r')
        else:
            bad_img.append((index, file_path, img.size))

    if bad_img:
        for b in bad_img:
            print("第{}张图片 <{}> 尺寸异常 - 尺寸为 {}".format(b[0], b[1], b[2]))
    else:
        print("未发现异常（共 {} 张图片）".format(len(img_list)))


def split(origin_dir, train_dir, test_dir):
    """
    分离训练集和测试集
    :return:
    """
    print("开始分离原始图片集为：测试集（5%）和训练集（95%）")

    # 图片名称列表和数量
    img_list = os.listdir(origin_dir)
    total_count = len(img_list)

    # 创建文件夹
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)

    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    # 测试集
    test_count = int(total_count*0.05)
    test_set = set()
    for i in range(test_count):
        while True:
            file_name = random.choice(img_list)
            if file_name in test_set:
                pass
            else:
                test_set.add(file_name)
                break

    test_list = list(test_set)
    print("测试集数量为：{}".format(len(test_list)))
    for file_name in test_list:
        src = os.path.join(origin_dir, file_name)
        dst = os.path.join(test_dir, file_name)
        shutil.move(src, dst)

    # 训练集
    train_list = os.listdir(origin_dir)
    print("训练集数量为：{}".format(len(train_list)))
    for file_name in train_list:
        src = os.path.join(origin_dir, file_name)
        dst = os.path.join(train_dir, file_name)
        shutil.move(src, dst)

    if os.listdir(origin_dir) == 0:
        print("migration done")


def main():
    # 图片路径
    origin_dir = sample_conf["origin_image_dir"]
    train_dir = sample_conf["train_image_dir"]
    test_dir = sample_conf["test_image_dir"]
    # 图片尺寸
    real_width = sample_conf["image_width"]
    real_height = sample_conf["image_height"]

    verify(origin_dir, real_width, real_height)
    split(origin_dir, train_dir, test_dir)


if __name__ == '__main__':
    main()

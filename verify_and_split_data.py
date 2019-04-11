"""
验证图片尺寸和分离测试集（5%）和训练集（95%）
初始化的时候使用，有新的图片后，可以把图片放在new目录里面使用。
"""
import json

from PIL import Image
import random
import os
import shutil


def verify(origin_dir, real_width, real_height, image_suffix):
    """
    校验图片大小
    :return:
    """
    if not os.path.exists(origin_dir):
        print("【警告】找不到目录{}，即将创建".format(origin_dir))
        os.makedirs(origin_dir)

    print("开始校验原始图片集")
    # 图片真实尺寸
    real_size = (real_width, real_height)
    # 图片名称列表和数量
    img_list = os.listdir(origin_dir)
    total_count = len(img_list)
    print("原始集共有图片: {}张".format(total_count))

    # 无效图片列表
    bad_img = []

    # 遍历所有图片进行验证
    for index, img_name in enumerate(img_list):
        file_path = os.path.join(origin_dir, img_name)
        # 过滤图片不正确的后缀
        if not img_name.endswith(image_suffix):
            bad_img.append((index, img_name, "文件后缀不正确"))
            continue

        # 过滤图片标签不标准的情况
        prefix, posfix = img_name.split("_")
        if prefix == "" or posfix == "":
            bad_img.append((index, img_name, "图片标签异常"))
            continue

        # 图片无法正常打开
        try:
            img = Image.open(file_path)
        except OSError:
            bad_img.append((index, img_name, "图片无法正常打开"))
            continue

        # 图片尺寸有异常
        if real_size == img.size:
            print("{} pass".format(index), end='\r')
        else:
            bad_img.append((index, img_name, "图片尺寸异常为：{}".format(img.size)))

    print("====以下{}张图片有异常====".format(len(bad_img)))
    if bad_img:
        for b in bad_img:
            print("[第{}张图片] [{}] [{}]".format(b[0], b[1], b[2]))
    else:
        print("未发现异常（共 {} 张图片）".format(len(img_list)))
    print("========end")
    return bad_img


def split(origin_dir, train_dir, test_dir, bad_imgs):
    """
    分离训练集和测试集
    :return:
    """
    if not os.path.exists(origin_dir):
        print("【警告】找不到目录{}，即将创建".format(origin_dir))
        os.makedirs(origin_dir)

    print("开始分离原始图片集为：测试集（5%）和训练集（95%）")

    # 图片名称列表和数量
    img_list = os.listdir(origin_dir)
    for img in bad_imgs:
        img_list.remove(img)
    total_count = len(img_list)
    print("共分配{}张图片到训练集和测试集，其中{}张为异常留在原始目录".format(total_count, len(bad_imgs)))

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
                img_list.remove(file_name)
                break

    test_list = list(test_set)
    print("测试集数量为：{}".format(len(test_list)))
    for file_name in test_list:
        src = os.path.join(origin_dir, file_name)
        dst = os.path.join(test_dir, file_name)
        shutil.move(src, dst)

    # 训练集
    train_list = img_list
    print("训练集数量为：{}".format(len(train_list)))
    for file_name in train_list:
        src = os.path.join(origin_dir, file_name)
        dst = os.path.join(train_dir, file_name)
        shutil.move(src, dst)

    if os.listdir(origin_dir) == 0:
        print("migration done")


def main():
    with open("conf/sample_config.json", "r") as f:
        sample_conf = json.load(f)

    # 图片路径
    origin_dir = sample_conf["origin_image_dir"]
    new_dir = sample_conf["new_image_dir"]
    train_dir = sample_conf["train_image_dir"]
    test_dir = sample_conf["test_image_dir"]
    # 图片尺寸
    real_width = sample_conf["image_width"]
    real_height = sample_conf["image_height"]
    # 图片后缀
    image_suffix = sample_conf["image_suffix"]

    for image_dir in [origin_dir, new_dir]:
        print(">>> 开始校验目录：[{}]".format(image_dir))
        bad_images_info = verify(image_dir, real_width, real_height, image_suffix)
        bad_imgs = []
        for info in bad_images_info:
            bad_imgs.append(info[1])
        split(image_dir, train_dir, test_dir, bad_imgs)


if __name__ == '__main__':
    main()

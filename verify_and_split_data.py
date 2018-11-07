from PIL import Image
import random
import os
import shutil


def verify():
    """
    校验图片大小
    :return:
    """
    bad_img = []
    print("Total image count: {}".format(len(img_list)))
    for index, img_name in enumerate(img_list):
        file_path = os.path.join(origin_dir, img_name)
        img = Image.open(file_path)

        if real_size == img.size:
            print("{} pass".format(index), end='\r')
        else:
            bad_img.append((index, file_path, img.size))

    for b in bad_img:
        print("第{}张图片 <{}> 尺寸异常 - 尺寸为 {}".format(b[0], b[1], b[2]))


def split():
    """
    分离训练集和测试集
    :return:
    """
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


if __name__ == '__main__':
    # 设置图片路径和正确的宽高
    origin_dir = "./sample/origin"
    train_dir = "./sample/train"
    test_dir = "./sample/test"
    real_width = 80
    real_height = 40
    real_size = (real_width, real_height)
    img_list = os.listdir(origin_dir)
    total_count = len(img_list)
    split()

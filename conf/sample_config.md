## 图片文件夹
```
origin_image_dir = "./sample/origin/"
new_image_dir = "./sample/new/"
train_image_dir = "./sample/train/"
test_image_dir = "./sample/test/"
api_image_dir = "./sample/api/"
online_image_dir = "./sample/online/"
local_image_dir = "./sample/local/"
```
## 模型文件夹
```
sample_conf.model_save_dir = "./model/"
```
## 图片相关参数
```
image_width = 100
image_height = 60
max_captcha = 4
image_suffix = "png"
```

## 是否从文件中的导入标签
```
use_labels_json_file = False
```

## 验证码字符相关参数
```
char_set = "0123456789abcdefghijklmnopqrstuvwxyz"
char_set = "abcdefghijklmnopqrstuvwxyz"
char_set = "0123456789"
```

## 在线识别远程验证码地址
```
remote_url = "http://127.0.0.1:6100/captcha/"
```

## 训练相关参数
```
# 到指定迭代次数后停止
cycle_stop = 3000
# 到指定准确率后停止
acc_stop = 0.99
# 每训练指定轮数就保存一次（覆盖之前的模型）
cycle_save = 500
# 使用GPU还是CPU,使用GPU需要安装对应版本的tensorflow-gpu==1.7.0
enable_gpu = 0
```
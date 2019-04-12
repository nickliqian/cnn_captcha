## 图片文件夹
```
origin_image_dir = "./sample/origin/"  # 原始文件
train_image_dir = "./sample/train/"   # 训练集
test_image_dir = "./sample/test/"   # 测试集
api_image_dir = "./sample/api/"   # api接收的图片储存路径
online_image_dir = "./sample/online/"  # 从验证码url获取的图片的储存路径
```
## 模型文件夹
```
model_save_dir = "./model/"  # 训练好的模型储存路径
```
## 图片相关参数
```
image_width = 80  # 图片宽度
image_height = 40  # 图片高度
max_captcha = 4  # 验证码字符个数
image_suffix = "jpg"  # 图片文件后缀
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
cycle_stop = 3000  # 到指定迭代次数后停止
acc_stop = 0.99  # 到指定准确率后停止
cycle_save = 500  # 每训练指定轮数就保存一次（覆盖之前的模型）
enable_gpu = 0  # 使用GPU还是CPU,使用GPU需要安装对应版本的tensorflow-gpu==1.7.0
```
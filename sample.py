from easydict import EasyDict

sample_conf = EasyDict()

# 图片文件夹
sample_conf.origin_image_dir = "./sample/origin/"
sample_conf.train_image_dir = "./sample/train/"
sample_conf.test_image_dir = "./sample/test/"
sample_conf.api_image_dir = "./sample/api/"
sample_conf.online_image_dir = "./sample/online/"

# 模型文件夹
sample_conf.model_save_dir = "./model/"

# 图片相关参数
sample_conf.image_width = 100
sample_conf.image_height = 60
sample_conf.max_captcha = 4
sample_conf.image_suffix = "png"

# 验证码字符相关参数
sample_conf.char_set = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
                        'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
# char_set = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
# char_set = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

sample_conf.remote_url = "https://www.xxxxx.com/getImg"

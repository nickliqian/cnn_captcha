from easydict import EasyDict

sample_conf = EasyDict()

sample_conf.origin_image_dir = "./sample/origin/"
sample_conf.train_image_dir = "./sample/train/"
sample_conf.test_image_dir = "./sample/test/"
sample_conf.api_image_dir = "./sample/api/"
sample_conf.online_image_dir = "./sample/online/"

sample_conf.image_width = 80
sample_conf.image_height = 40
sample_conf.max_captcha = 4
sample_conf.image_suffix = "jpg"

sample_conf.char_set = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
                        'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
# char_set = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
# char_set = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

sample_conf.remote_url = "https://xin.baidu.com/check/getCapImg?t=1534261894187"

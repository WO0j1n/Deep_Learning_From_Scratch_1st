import sys, os
sys.path.append(os.pardir)
import numpy as np
from PIL import Image
from mnist import load_mnist

def img_show(img):
  pil_img = Image.fromarray(np.uint8(img))
  pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
# flatten을 True로 설정하여 다차원 배열을 1차원 배열로 바꿈
# normalize을 False 정규화를 하지 않음을 알 수 있다. (0.0 ~ 1.0 사이로)

img = x_train[0]
label = t_train[0]
print(label)


print(img.shape)
img = img.reshape(28, 28) # 이미지를 원래 모양으로 바꿔줌
print(img.shape)

img_show(img)
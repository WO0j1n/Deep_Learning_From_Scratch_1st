def step_function(x):
    if x > 0:
      return 1
    else:
      return 0
# 이 경우 x는 실수만 가능하고 넘파이 배열을 인수로 넣을 수 없음

# 실수를 넘어서 넘파이 배열을 인수로 넣을 수 있는 개선된 코드
import numpy as np

def step_function(x):
  y = x > 0
  return y.astype(np.int)

import numpy as np

# 예제 데이터 생성
x = np.array([-1.0, 1.0, 2.0])
print(x)

y = x > 0
print(y)

# astype 메서드를 사용할 때 int로 변환 -> () 괄호 안에 내가 변환하고 싶은 자료형 써주기
y = y.astype(int)
print(y)

import numpy as np
import matplotlib.pyplot as plt

def step_function(x):
  return np.array(x > 0, dtype = int) # bool 값을 int형으로 변환하여 넘파이 배열로 저장

x = np.arange(-5.0, 5.0, 0.1) # -5.0 ~ 5.0 전까지 0.1 간격으로 넘파이 배열 생성 => -5.0 ~ 4.9
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1) # y축 범위 지정
plt.show()

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

import numpy as np

x = np.array([-1.0, 1.0, 2.0])
print(sigmoid(x))

t = np.array([1.0, 2.0, 3.0])

print(1.0 + t)

print(1.0 / t)

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

def ReLU(x):
  return np.maximum(0, x) # maximum() => 두 입력 중 큰 값을 선택해 반환하는 함수

import numpy as np
A = np.array([1,2,3,4])
print(A)

print(np.ndim(A)) # 차원

print(A.shape) # 형태(행, 열)
print(A.shape[0]) # 행의 개수

B = np.array([[1,2], [3,4], [5,6]])
print(B)

print(np.ndim(B)) # 차원

print(B.shape) # 형태(행, 열)
print(B.shape[0]) # 행의 개수
print(B.shape[1]) # 열의 개수

A = np.array([[1,2], [3,4]])
print(A.shape)

B = np.array([[5, 6], [7, 8]])
print(B.shape)

print(np.dot(A, B))
print(A @ B)

# C = np.array([[1,2], [3,4]])
# print(C.shape)
#
# A = np.array([[1,2,3], [4,5,6]])
# print(A.shape)
#
# print(np.dot(A, C))
# print(A @ C)
# 오류 발생 코드 -> 1번째 차원의 열의 원소 수, 2번째 차원의 행의 원소 수 가 동일 해야 함. 즉, 차원이 맞지 않아서 수행 불가능

A = np.array([[1,2], [3,4], [5,6]])
print(A.shape)

B = np.array([7, 8])
print(B.shape)

print(np.dot(A, B))
print(A @ B)

# 신경망에서는 행령의 곱으로 계산을 수행한다. => 가중치 * 입력신호
X = np.array([1,2])
print(X.shape)

W = np.array([[1,3,5], [2,4,6]])
print(W.shape)

Y = np.dot(X, W)
print(Y)

X = np.array([1., 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

print(W1.shape)
print(X.shape)
print(B1.shape)

A1 = np.dot(X, W1) + B1
print(A1)

X = np.array([1., 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

print(W1.shape)
print(X.shape)
print(B1.shape)

A1 = np.dot(X, W1) + B1
print(A1)

Z1 = sigmoid(A1)
print(Z1)

W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

print(W2.shape)
print(Z1.shape)
print(B2.shape)

A2 = np.dot(Z1, W2) + B2
print(A2)

Z2 = sigmoid(A2)
print(Z2)

def identity_function(x):
  return x

W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

print(W3.shape)
print(Z2.shape)
print(B3.shape)

A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3)

print(Y)

def init_network(): # 가중치와 편향을 딕셔너리에 저장
  network = {}

  network['W1']  = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
  network['b1']  = np.array([0.1, 0.2, 0.3])
  network['W2']  = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
  network['b2']  = np.array([0.1, 0.2])
  network['W3']  = np.array([[0.1, 0.3], [0.2, 0.4]])
  network['b3']  = np.array([0.1, 0.2])

  return network

def forward(network, x): # 순방향 전파로 각 숫자는 각 층에서의 연산과정임.
  W1, W2, W3 = network['W1'], network['W2'], network['W3']
  b1, b2, b3 = network['b1'], network['b2'], network['b3']

  a1 = np.dot(x, W1) + b1
  z1 = sigmoid(a1)
  a2 = np.dot(z1, W2) + b2
  z2 = sigmoid(a2)
  a3 = np.dot(z2, W3) + b3
  y = identity_function(a3)

  return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)

# 소트프맥스 함수 구현 -> 분류의 경우, 출력층 활성화 함수
# 항등 함수 -> 회귀의 경우, 출력층 활성화 함수

a = np.array([0.3, 2.9, 4.0])

exp_a = np.exp(a)
print(exp_a)

sum_exp_a = np.sum(exp_a)
print(sum_exp_a)

y = exp_a / sum_exp_a
print(y)

def softmax(a):
  exp_a = np.exp(a)
  sum_exp_a = np.sum(exp_a)
  y = exp_a / sum_exp_a

  return y
# 위에서 구현한 두가지의 소프트맥스 함수는 지수함수의 특성에 의해 오버플로우 연산이 발생 -> 이를 방지하고자 임의의 정수 C에 log를 취하여 더해주거 빼준 값을 지수함수로 써서 해결한다.
# 이때 임의의 정수 C는 입력 신호 중 가장 큰 신호를 사용한다.

a = np.array([1010, 1000, 990])

# 해당 부분에서 오류가 있다고 주장하는 것은 오버플로가 발생했음을 알려주는 것이다. 그러나 우리는 임이의 정수 C 즉, 입력 신호 중 가장 큰 신호를 로그를 통해서 더하거나 뺴주는 과정을 보고자 이렇게 표현한 것임.
print(np.exp(a) / np.sum(np.exp(a)))

c = np.max(a)
print(c)

print(np.exp(a - c) / np.sum(np.exp(a - c)))

def softmax(a):
  c = np.max(a)

  exp_a = np.exp(a-c)
  sum_exp_a = np.sum(exp_a)

  y = exp_a / sum_exp_a

  return y


# 소프트맥스 함수의 출력은 0에서 1.0 사이의 실수  -> 소프트맥스 함수 출력의 총합은 항상 1임을 볼 수 있다!!!!
# 소프트맥스 함수의 출력의 총합이 항상 1이기에 이를 '확률'로 해석이 가능함. => 소수점 2개 당겨 오기
# 소프트 맥스 함수에서 입력 신호의 대소 관계는 출력에서의 대소 관계로 이어짐.
# 분류 - 소프트맥스 함수(지수 함수에서의 계산 자원 낭비를 줄이고자 생략하기도 함. 단, 추론의 경우만(매개변수 자동 학습 제거한 신경망)), 회귀 -항등함수

a = np.array([0.3, 2.9, 4.0])
y = softmax(a)

print(y)

print(np.sum(y))


import os
import pickle
import numpy as np
from tensorflow.keras.datasets import mnist

# MNIST 데이터셋을 로드합니다.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 데이터셋의 형태를 출력합니다.
print("Training data shape:", x_train.shape)
print("Test data shape:", x_test.shape)
#위에 오류 코드는 저자의 깃허브에서 가져온 코드로 왜 오류가 발생했는지는 모르겠다...
dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/mnist.pkl"


def download_mnist():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    dataset = {
        'train_img': train_images.reshape(-1, 28 * 28),
        'train_label': train_labels,
        'test_img': test_images.reshape(-1, 28 * 28),
        'test_label': test_labels
    }
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("MNIST dataset downloaded and saved as pickle.")


def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """MNIST 데이터셋 읽기

    Parameters
    ----------
    normalize : 이미지의 픽셀 값을 0.0~1.0 사이의 값으로 정규화할지 정한다.
    one_hot_label :
        one_hot_label이 True면、레이블을 원-핫(one-hot) 배열로 돌려준다.
        one-hot 배열은 예를 들어 [0,0,1,0,0,0,0,0,0,0]처럼 한 원소만 1인 배열이다.
    flatten : 입력 이미지를 1차원 배열로 만들지를 정한다.

    Returns
    -------
    (훈련 이미지, 훈련 레이블), (시험 이미지, 시험 레이블)
    """
    if not os.path.exists(save_file):
        download_mnist()

    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if one_hot_label:
        dataset['train_label'] = _change_ont_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_ont_hot_label(dataset['test_label'])

    if not flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])


def _change_ont_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
    return T


if __name__ == '__main__':
    download_mnist()

import sys, os
sys.path.append(os.pardir) # 부모 디렉터리의 파일을 가져올 수 있도록 설정함
from mnist import load_mnist


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False) # 다차원 데이터를 1차원으로 변경하였고 정규화는 하지 않음.

# normalize, flatten, one_hot_label
# normalize: 정규화 -> 0.0 ~ 1.0 사이의 값으로 정규화
# flatten: 평탄화 -> 다차원 배열을 1차원 배열로 만듦
# one_hot_label: 정답을 뜻하는 원소만 1, 마너지는 모두 0인 배열로 설정, 즉 True을 1로 False는 0으로 => 원-핫

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)

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



import pickle
import numpy as np

def create_sample_weight():
    network = {
        'W1': np.random.rand(784, 50),
        'b1': np.random.rand(50),
        'W2': np.random.rand(50, 100),
        'b2': np.random.rand(100),
        'W3': np.random.rand(100, 10),
        'b3': np.random.rand(10)
    }
    with open("/ch03/sample_weight.pkl", 'wb') as f:
        pickle.dump(network, f)

create_sample_weight()

import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.pardir) # 부모 디렉터리의 파일을 가져올 수 있도록 설정함
from mnist import load_mnist

def get_data():
  (x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize = True, flatten = True, one_hot_label = False)
  return x_test, t_test

def init_network():
  with open("sample_weight.pkl", 'rb') as f:
    network = pickle.load(f)
  return network

def predict(network,x):
  W1, W2, W3 = network['W1'], network['W2'], network['W3']
  b1, b2, b3 = network['b1'], network['b2'], network['b3']

  a1 = np.dot(x, W1) + b1
  z1 = sigmoid(a1)
  a2 = np.dot(z1, W2)  + b2
  z2 = sigmoid(a2)
  a3 = np.dot(z2, W3)  + b3
  y = softmax(a3)

  return y


x,  t = get_data()
network = init_network()

accuracy_cnt = 0
for i in range(len(x)):
  y = predict(network, x[i])
  p = np.argmax(y)

  if p == t[i]:
    accuracy_cnt += 1

print("Accuracy: " + str(float(accuracy_cnt)/len(x)))

x, _ = get_data()

network = init_network()
W1, W2, W3 = network['W1'], network['W2'],  network['W3']

print(x.shape)
print(x[0].shape)
print(W1.shape)
print(W2.shape)
print(W3.shape)


x, t = get_data()
network = init_network()

batch_size = 100
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i: i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i: i+batch_size])

print("Accuracy: "+ str(float(accuracy_cnt)/ len(x)))


list(range (0, 10))
list(range(0, 10, 3))

x = np.array([[0.1, 0.8, 0.1], [0.3, 0.1, 0.6], [0.2, 0.5, 0.3], [0.8, 0.1, 0.1]])
y = np.argmax(x, axis = 1)
print(y)


y = np.array([1,2,1,0])
t = np.array([1,2,0,0])
print(y==t)

print(np.sum(y==t))
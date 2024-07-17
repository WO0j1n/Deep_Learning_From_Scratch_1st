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

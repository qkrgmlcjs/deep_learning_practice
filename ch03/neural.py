# %%
def step_function(x) :
    #if x>0 :
       # return 1
   # else :
      #  return 0
    y = x>0
    return y.astype(np.int)

# %%
import numpy as np
# %%
x= np.array([-1.0,1.0,2.0])
# %%
x
# %%
y = x>0 #넘파이 배열에 부등호 연산을 수행하면 배열의 원소 각각에 부등호 연산 수행
        # bool배열이 생성된다
# %%
y
# %%
y= y.astype(np.int64) # 넘파이 배열의 자료형을 변환할 때는 astype()메소드 이용함.
# %%
y
# %%
import matplotlib.pylab as plt
# %%
def step_function (x) :
    return np.array(x>0,dtype = np.int64)

x = np.arange(-5.0,5.0,0.1) # -5.0 ~ 5.0 사이의 실수를 0.1간격으로 배열 생성
y = step_function(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1)
plt.show()

# %%
def sigmoid(x) :
    return 1 /(1+np.exp(-x))
# %%
x = np.array([-1.0,1.0,2.0])
sigmoid(x)
# %%
t= np.array([1.0,2.0,3.0])
1.0+t
# %%
1.0/t
# %%
x = np.arange(-5.0,5.0,0.1) # -5.0 ~ 5.0 사이의 실수를 0.1간격으로 배열 생성
y = sigmoid(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1)
plt.show()
# %%
def relu (x): #시그모이드 말고 요즘 신경망에서 자주 쓰는 함수라고 함.
    return np.maximum(0,x) #넘파이의 maximum함수 두 입력 중 큰값을 선택해 반환하는 함수
# %%
import numpy as np
A = np.array([1,2,3,4])
print(A)
np.ndim(A) #dimension -배열의 차원 출력
A.shape #튜플을 반환함
#A.shape[0]
# %%
B=np.array([[1,2],[3,4],[5,6]])
print(B)
# %%
np.ndim(B)
# %%
B.shape
# %%
A=np.array([[1,2],[3,4]])
A.shape
# %%
B=np.array([[5,6],[7,8]])
B.shape
# %%
np.dot(A,B) #행렬의 곱 dot() (스칼라 곱) dot product라고 합니다.
# %%
A = np.array([[1,2,3],[4,5,6]])
A.shape
# %%
B = np.array([[1,2],[3,4],[5,6]])
# %%
B.shape
# %%
np.dot(A,B)
# %%
C = np.array([[1,2],[3,4]])
C.shape
# %%
np.dot(A,C)
# %%
A = np.array([[1,2],[3,4],[5,6]])
A.shape

# %%
B = np.array([7,8])
# %%
np.dot(A,B)
# %%
X = np.array([1,2])
X.shape
print(X)
# %%
W = np.array([[1,3,5],[2,4,6]])
print(W)
# %%
W.shape


# %%
Y = np.dot(X,W)
# %%
print(Y)
# %%
import numpy as np
X = np.array([1.0,0.5])
W1 = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
B1 = np.array([0.1,0.2,0.3])

print(W1.shape)
print(X.shape)
print(B1.shape)
# %%
A1 = np.dot(X,W1) + B1
# %%
print(A1)
# %%
Z1 = sigmoid(A1)

print(A1)
print(Z1)
# %%
W2 = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
B2 = np.array([0.1,0.2])

print(Z1.shape)
print(W2.shape)
print(B2.shape)
# %%
A2 = np.dot(Z1,W2) + B2
Z2 = sigmoid(A2)
# %%
print(A2)
print(Z2)
# %%
def identify_function(x) :
    return X

W3 = np.array([[0.1,0.3],[0.2,0.4]])
B3 = np.array([0.1,0.2])

A3 = np.dot(Z2,W3) +B3
print(A3)
Y = identify_function(A3) # Y=A3
print(Y)
#출력층 함수는 성질에 맞게 사용
#회귀에는 항등함수, 2클래스 분류에는 시그모이드 함수, 다중클래스 분류에는 소프트맥스 함수
# %%
def init_network() :
    network ={} #딕셔너리 자료형
    network['W1'] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['b1'] = np.array([0.1,0.2,0.3])
    network['W2'] = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['b2'] = np.array([0.1,0.2])
    network['W3'] = np.array([[0.1,0.3],[0.2,0.4]])
    network['b3'] = np.array([0.1,0.2])
    
    return network

def  forward(network,x):
    W1,W2,W3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']
    a1 = np.dot(x,W1) +b1
    z1 = sigmoid(a1)
    print( z1)
    a2 = np.dot(z1,W2) +b2
    z2 = sigmoid(a2)
    print( z2)
    a3 = np.dot(z2,W3) +b3
    y = identify_function(a3)
    print( y)
    return y

network = init_network()
print(network)
x = np.array([1.0,0.5])
y = forward(network,x)
print(y)    
# %%

# %%
# softmax함수
import numpy as np
a= np.array([0.3,2.9,4.0])
# %%
def softmax(a) :
    c = np.max(a) #오버플로를 위한 대책 상쇄하는 효과
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a/sum_exp_a
    
    return y
# %%
a = np.array([0.3,2.9,4.0])
y = softmax(a)
# %%
print(y)
# %%
np.sum(y)
# %%
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
# %%
(x_train,t_train),(x_test,t_test) = \
    load_mnist(flatten=True, normalize=False)
#load_mnist 함수는 읽은 MNIST 데이터를 "(훈련 이미지, 훈련 레이블),(시험 이미지, 시험 레이블)"형식으로 반환
#인수로는 normalize, flatten, one_hot_label, 세가지 다 bool형식
#normalize 입력 이미지의 픽셀 값을 0.0에서 1.0사이의 값으로 정할것인지 T/F F면 0~255유지
#flatten 입력이미지 평탄(1차원으로 바꿀것인지) T ->1차원 배열, F->3차원 배열
#one-hot encoding [0,0,0,1,0] 처럼 정답을 뜻하는 것만 1이고 나머진 0, false면 7이나 2같이 숫자 형태의 레이블 저장
# %%
print(x_train.shape)
# %%
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)
# %%

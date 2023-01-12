#%%
import sys, os
sys.path.append(os.pardir)
import numpy as np
import pickle
from dataset.mnist import load_mnist
from PIL import Image

def softmax(a) :
    c = np.max(a) #오버플로를 위한 대책 상쇄하는 효과
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a/sum_exp_a
    
    return y
def sigmoid(x) :
    return 1 /(1+np.exp(-x))

def get_data() :
    (x_train,t_train),(x_test,t_test) = \
        load_mnist(flatten=True, normalize=True,one_hot_label = False)
    return x_test,t_test

def init_network():
    with open("sample_weight.pkl",'rb') as f: #파일 여는것임
        network = pickle.load(f)# 이 파일에는 학습된 가중치 매개변수 딕셔너리 형태로 저장
        
    return network

def predict(network, x) :
    W1,W2,W3 = network['W1'], network['W2'], network['W3'] 
    b1,b2,b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x,W1) +b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2) +b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3) +b3
    y= softmax(a3)
    return y
    
# %%

x,t = get_data()
network = init_network()
accuracy_cnt = 0
for i in range(len(x)) :
    y = predict(network,x[i]) #각 레이블의 확률을 넘파이 배열로 반환
    p = np.argmax(y) #확률이 가장 높은 값 구한다 ->예측결과
    if p==t[i] :
        accuracy_cnt += 1
    
# 0-255를 0.0~1.0 이렇게 바꾸는걸 정규화, 입력데이터에 특정변환을 가하는 것을 전처리;
print ("Accuracy" +str(float(accuracy_cnt)/len(x)))
# %%
x,t = get_data()
network = init_network()

batch_size = 100
accuracy_cnt = 0

for i in range(0,len(x),batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network,x_batch)
    p=np.argmax(y_batch,axis=1)
    accuracy_cnt += np.sum(p==t[i:i+batch_size])
    
print ("Accuracy" +str(float(accuracy_cnt)/len(x)))
# %%
#손실함수 -> 평균제곱오차, 교차엔트로피
def mean_squared_error(y,t) :
    return 0.5 * np.sum((y-t)**2) # 평균제곱오차 공식



# %%
import numpy as np
t= [0,0,1,0,0,0,0,0,0,0]

y=[0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]
mean_squared_error(np.array(y),np.array(t))
# %%
y=[0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0]
mean_squared_error(np.array(y),np.array(t))
# %%
def cross_entropy_error(y,t) :
    delta = 1e-7
    return -np.sum(t*np.log(y+delta))
# %%
t= [0,0,1,0,0,0,0,0,0,0]
y=[0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]
cross_entropy_error(np.array(y),np.array(t))
# %%

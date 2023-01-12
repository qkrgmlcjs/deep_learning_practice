#and
#%%
def AND (x1,x2) :
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1*w1 + x2*w2
    if tmp <= theta :
        return 0
    elif tmp >theta :
        return 1
    
# %%
import numpy as np

x= np.array([0,1]) # 입력
w= np.array([0.5,0.5]) #  가중치
b=-0.7 
#넘파이에서 array의 개수가 각각 같다면 곱하면 각 원소끼리 곱합니다..
# %%
import numpy as np
def AND(x1,x2) :
    x= np.array([x1,x2]) # 입력
    w= np.array([0.5,0.5]) #  가중치
    b=-0.7 
    tmp = np.sum(w*x) +b
    if tmp <= 0 :
        return 0
    elif tmp > 0 :
        return 1
# %%
def NAND(x1,x2) :
    x = np.array([x1,x2])
    w = np.array([-0.5,-0.5])
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0 :
        return 0 
    if tmp > 0:
        return 1 
    
def OR(x1,x2) : 
    x=np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.2
    tmp = np.sum(w*x) +b
    if tmp <=0 :
        return 0
    elif tmp > 0 :
        return 1
    
# %%
def XOR(x1,x2):
    s1 = NAND(x1,x2)
    s2 = OR(x1,x2)
    y = AND(s1,s2)
    return y
    
# %%

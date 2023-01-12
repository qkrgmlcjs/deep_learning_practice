#%%
def numerical_diff(f,x) :
    h = 1e-4
    return (f(x+h)-f(x-h)) /(2*h)
# %%
def function_1(x):
    return 0.01 * x **2 + 0.1*x

# %%
import numpy as np
import matplotlib.pylab as plt

x = np.arange(0.0,20.0,0.1) # 0에서 20까지 0.1 간격 배열 x
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x,y)
plt.show()
# %%
numerical_diff(function_1, 5)
# %%
numerical_diff(function_1,10)
# %%
def function_2 (x) :
    return x[0]**2 + x[1] **2
# %%
def numerical_gradient(f,x) :
    h = 1e-4
    grad = np.zeros_like(x) # x와 형상이 같은 배열 생성
    
    for idx in range(x.size) : 
        tmp_val = x[idx] #f(x+h) 계산
        x[idx] = tmp_val+h
        fxh1 = f(x)
        
        x[idx] = tmp_val-h
        fxh2 = f(x)
        
        grad[idx] = (fxh1 - fxh2) /(2*h)
        x[idx] = tmp_val
        
    return grad
# %%

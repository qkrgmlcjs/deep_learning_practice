#%%
class SGD: #SGD는 최적화 기법이다
    def __init__ (self,lr = 0.01): #lr = learning rate
        self.lr = lr
        
    def update (self, params, grads) : # sgd과정에서 반복해서 불림 params와 grads는 딕셔너리 변수이다. 늘 그랬던것 처럼 ex)params['W1'],grads['W1']
        for key in params.key():
            params[key] -= self.lr * grads[key]
# %%

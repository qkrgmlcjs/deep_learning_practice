import imp
#%%

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0,6,0.1) # 0에서 6까지 0.1간격으로 생성
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x,y1, label = "sin")
plt.plot(x,y2,linestyle = "--",label = "cos")
plt.xlabel("x")
plt.ylabel("y")
plt.title('sin&cos')
plt.legend()
plt.show()
# %%
import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('lena.png')

plt.imshow(img)
plt.show
# %%

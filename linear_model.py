import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


data = np.loadtxt('train_data.txt', delimiter=',')
data = data.tolist()
for i in range(len(data)):
    data[i] = tuple(data[i])
data = set(data)
data = list(data)
data = np.asarray(data)
print("data")
print(data)
# print(data.shape)
# plt.scatter(data[:,0],data[:,1])
# plt.xlabel("real value")
# plt.ylabel("expected value")
# plt.show()
X = data[:,0].reshape(-1,1)
y = data[:,1].reshape(-1,1)

reg = LinearRegression().fit(X,y)

print("predicted data")
print(reg.predict(X).reshape(1,-1))
x_plot = np.linspace(15,80,50).reshape(-1,1)
y_plot = reg.predict(x_plot)

print(data.shape)
plt.scatter(data[:,0],data[:,1])
plt.plot(x_plot,y_plot,'r')
plt.xlabel("real value")
plt.ylabel("expected value")
plt.show()






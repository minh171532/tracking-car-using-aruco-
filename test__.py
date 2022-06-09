import numpy as np


a = np.array([[1,1,1]])
b = np.array([[2,2,2]])
c = np.array([[3,3,3]])
a = a.reshape((1,1,3))
b = b.reshape((1,1,3))
c = c.reshape((1,1,3))

d = np.concatenate((a[0],b[0],c[0]))
print(d)

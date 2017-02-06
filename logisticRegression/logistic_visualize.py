import numpy as np
import matplotlib.pyplot as plt

#samples
N = 100
#feature for each sample
D = 2

X = np.random.randn(N,D)

# Center the first 50 points at (-2,-2)
X[:50,:] = X[:50,:] - 2*np.ones((50,D))

# Centerthe last 50 points at (2,2)
X[50:,:] = X[50:,:] + 2*np.ones((50,D))

# Labels: first 50 are 0, last 50 are 1
T = np.array([0]*50 + [1]*50)

# add a column of ones
ones = np.array([[1]*N]).T
Xb = np.concatenate((ones,X), axis=1)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

# get the closed-form solution
w = np.array([0,4,4])

# y = -x

# alpha = 0.5 means the dots are transparent
plt.scatter(X[:,0],X[:,1],c = T,s= 100,alpha=0.5)

x_axis = np.linspace(-6,6,100)
y_axis = -x_axis
plt.plot(x_axis,y_axis)
plt.show()

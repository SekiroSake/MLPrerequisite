import numpy as np
import matplotlib.pyplot as plt

N = 50

X = np.linspace (0,10,N)
Y = 0.5*X + np.random.randn(N)

# last one in the array
Y[-1] += 30
# second last one in the array
Y[-2] += 30

plt.scatter(X,Y)
plt.show()

X = np.vstack([np.ones(N),X]).T
w_ml = np.linalg.solve(X.T.dot(X) , X.T.dot(Y))
Yhat_ml = X.dot(w_ml)
plt.scatter(X[:,1],Y)
plt.plot(X[:,1],Yhat_ml)
plt.show()

l2 = 1000.0
w_map = np.linalg.solve(l2*np.eye(2) + X.T.dot(X), X.T.dot(Y))
Yhat_map = X.dot(w_map)
plt.scatter(X[:,1],Y)
plt.plot(X[:,1],Yhat_ml,label = 'maximum likehood')
plt.plot(X[:,1],Yhat_map,label = 'map')
plt.legend()
plt.show()

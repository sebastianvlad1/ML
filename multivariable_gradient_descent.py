import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def add_bias(x):
    return np.column_stack((np.ones(x.shape[0]), x))

def normalization(x):
    std = np.std(x)
    mean = np.mean(x)
    x_normalized = (x - mean) / std
    return x_normalized

def gradient_descent(x, y, w, learning_rate, iterations):
    m = len(y)
    for i in range(iterations):
        predictions = x.dot(w)
        error = predictions - y
        dldw = (1 / m) * x.T.dot(error)
        w = w - learning_rate * dldw
        cost = (1/(2*m)) * np.sum((x.dot(w) - y) ** 2)
        print(cost)
    return w


np.random.seed(42)

x = 1.5 * np.random.rand(100, 3)

y = 2 + x.dot(np.array([2, 1.6, 0.3])) + np.random.randn(100)

x_normalized = normalization(x)

x = add_bias(x_normalized)

learning_rate = 0.01

iterations = 1000

w = np.zeros(x.shape[1])

w = gradient_descent(x, y, w, learning_rate, iterations)

print(w)

y_pred = x.dot(w)

r2_score = r2_score(y, y_pred)

print("r2 score: ", r2_score)

plt.plot(y, y_pred, 'o')
plt.show()





import numpy as np
import matplotlib.pyplot as plt

# Initialise parameters
# x = np.random.randn(10, 1)
# y = 2*x + np.random.rand()

x = np.array([32.50234527, 53.42680403, 61.53035803, 47.47563963, 59.81320787,
           55.14218841, 52.21179669, 39.29956669, 48.10504169, 52.55001444,
           45.41973014, 54.35163488, 44.1640495 , 58.16847072, 56.72720806,
           48.95588857, 44.68719623, 60.29732685, 45.61864377, 38.81681754])
y = np.array([31.70700585, 68.77759598, 62.5623823 , 71.54663223, 87.23092513,
           78.21151827, 79.64197305, 59.17148932, 75.3312423 , 71.30087989,
           55.16567715, 82.47884676, 62.00892325, 75.39287043, 81.43619216,
           60.72360244, 82.89250373, 97.37989686, 48.84715332, 56.87721319])

# Parameters
w = 0.0
b = 0.0

#Hyperparameter
learning_rate = 0.00001

# Gradient descent func
def gradient_descent(x, y, w, b, learning_rate):
    # calc derivatives of loss with respect ot the parameters
    dldw = 0.0
    dldb = 0.0

    for xi, yi in zip(x, y):
        dldw += -2*xi*(yi-(w*xi+b)) # chain rule
        dldb += -2*(yi-(w*xi+b))

    w = w - learning_rate * dldw
    b = b - learning_rate * dldb
    return w, b


# Iteratively make updates
for epoch in range(700):
    w, b = gradient_descent(x, y, w, b, learning_rate)
    print(w, b)
    ssr = np.sum((y - (w*x + b))**2)
    print(ssr)
    # plt.scatter(w, ssr, color="red")
    # plt.scatter(b, ssr, color="blue")
    # plt.pause(0.001)
    # plt.clf()
    pass

y_values_f1 = y
y_values_f2 = w*x + b

# Plot the functions
plt.scatter(x, y_values_f1, label='data points')
plt.plot(x, y_values_f2, label='linear regression')

# Add labels and title
plt.xlabel('x')
plt.ylabel('y')
plt.title('Liniar Regression')
plt.legend()
plt.show()

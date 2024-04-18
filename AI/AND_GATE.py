import numpy as np
import matplotlib.pyplot as plt
def TLU(x) :
    if x >= 0 :
      return 1.0
    return 0

def sigmoid(x) :
    return (1.0 / (1.0 + np.exp(-x)))

def generate_surface(weights):
    X = np.linspace(0, 1, 121)
    Y = np.linspace(0, 1, 121)
    X, Y = np.meshgrid(X, Y)
    Z = sigmoid(weights[2] * X + weights[1] * Y + weights[0])
    return X, Y, Z


x  = np.array([
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [1.0, 1.0, 1.0]])
t  = np.array([0.0, 1.0, 1.0, 1.0])

lrate = 0.1

w = np.zeros(3)

w[0] = 0.5
w[1] = 0.2
w[2] = 0.7

for epoch in range (10000):
    print("epoch", epoch)
    for i in range(4):
        out = w[0]*x[i][0] + w[1]*x[i][1] + w[2]*x[i][2]
        y = sigmoid(out)

        print(t[i], y, out)
        #print(w)

        err = t[i] - y

        for j in range(3) :
            w[j] = w[j] + lrate * err * (y * (1 - y))* x[i][j]


X, Y, Z = generate_surface(w)
plt.figure()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()

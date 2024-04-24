import numpy as np

TData = np.array([[0.9, 0.9, 0.9, 0.9, 0.9],
               [0.1, 0.1, 0.9, 0.1, 0.1],
               [0.1, 0.1, 0.9, 0.1, 0.1],  # T
               [0.1, 0.1, 0.9, 0.1, 0.1],
               [0.1, 0.1, 0.9, 0.1, 0.1]])

CData = np.array([[0.9, 0.9, 0.9, 0.9, 0.9],
               [0.9, 0.1, 0.1, 0.1, 0.1],
               [0.9, 0.1, 0.1, 0.1, 0.1],  # C
               [0.9, 0.1, 0.1, 0.1, 0.1],
               [0.9, 0.9, 0.9, 0.9, 0.1]])

EData = np.array([[0.9, 0.9, 0.9, 0.9, 0.1],
               [0.9, 0.1, 0.1, 0.1, 0.1],
               [0.9, 0.9, 0.9, 0.9, 0.9],  # E
               [0.9, 0.1, 0.1, 0.1, 0.1],
               [0.9, 0.9, 0.9, 0.9, 0.1]])

GData = np.array([[0, 1, 1, 1, 1],
                  [1, 0, 0, 0, 0],
                  [1, 0, 0, 1, 1],
                  [1, 0, 0, 0, 1],
                  [0, 1, 1, 1, 1]])

BData = np.array([[1, 1, 1, 1, 1],
                  [1, 0, 0, 0, 1],
                  [1, 1, 1, 1, 1],
                  [1, 0, 0, 0, 1],
                  [1, 1, 1, 1, 1]])

a = TData.sum(axis=1)

b = CData.sum(axis=1)

c = EData.sum(axis=1)

d = TData.sum(axis=0)

e = CData.sum(axis=0)

f = EData.sum(axis=0)
learning_data_X = np.zeros((3, 5))
learning_data_Y = np.zeros((3, 5))
learning_data_X[0] = a
learning_data_X[1] = b
learning_data_X[2] = c

learning_data_Y[0] = d
learning_data_Y[1] = e
learning_data_Y[2] = f

print(learning_data_X)
print(learning_data_Y)

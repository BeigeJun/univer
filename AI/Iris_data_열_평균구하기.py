import numpy as np
from sklearn.datasets import load_iris
import pandas as pd

def avr(A,B):
    sum = 0
    for i in range(len(A)):
        sum += A[i][B]
    output = sum / len(A)
    return output

Iris = load_iris()
Iris_Data = pd.DataFrame(data=Iris['data'], columns=Iris['feature_names'])
iris_data = np.array([])
iris_data = np.array(Iris_Data.iloc[:])
print(iris_data.shape)
print(avr(iris_data,0))
print(avr(iris_data,1))
print(avr(iris_data,2))
print(avr(iris_data,3))

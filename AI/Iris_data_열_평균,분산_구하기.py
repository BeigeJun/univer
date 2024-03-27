import numpy as np
from sklearn.datasets import load_iris
import pandas as pd

def avr(A,B):
    sum = 0
    for i in range(len(A)):
        sum += A[i][B]
    output = sum / len(A)
    return output

def variance(A, B):
    avg = avr(A, B)
    diff = 0
    for i in range(len(A)):
        diff += (A[i][B] - avg) ** 2
    variance_output = diff / len(A)
    return variance_output

Iris = load_iris()
Iris_Data = pd.DataFrame(data=Iris['data'], columns=Iris['feature_names'])
iris_data = np.array([])
iris_data = np.array(Iris_Data.iloc[:])
print(iris_data.shape)
print("Average Sepal Length:", avr(iris_data,0))
print("Variance Sepal Length:", variance(iris_data,0))
print("Average Sepal Width:", avr(iris_data,1))
print("Variance Sepal Width:", variance(iris_data,1))
print("Average Petal Length:", avr(iris_data,2))
print("Variance Petal Length:", variance(iris_data,2))
print("Average Petal Width:", avr(iris_data,3))
print("Variance Petal Width:", variance(iris_data,3))

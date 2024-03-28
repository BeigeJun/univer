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
iris_data_ALL = np.array(Iris_Data.iloc[:])
iris_data_A = np.array(Iris_Data.iloc[:49])
iris_data_B = np.array(Iris_Data.iloc[50:99])
iris_data_C = np.array(Iris_Data.iloc[100:149])
iris_data_1 = np.array(Iris_Data.iloc[:29])
iris_data_2 = np.array(Iris_Data.iloc[30:59])
iris_data_3 = np.array(Iris_Data.iloc[60:89])
iris_data_4 = np.array(Iris_Data.iloc[90:119])
iris_data_5 = np.array(Iris_Data.iloc[120:149])

def make_chart(data):
  a1=avr(data,0)
  a2=avr(data,1)
  a3=avr(data,2)
  a4=avr(data,3)
  a5=variance(data,0)
  a6=variance(data,1)
  a7=variance(data,2)
  a8=variance(data,3)
  chart = pd.DataFrame({'ID':['Average Sepal', 'Variance'],
                             'Sepal Length':[a1, a5],
                             'Sepal Width':[a2, a6],
                             'Petal Length':[a3,a7],
                             'Petal Width':[a4,a8]})
  print(chart)
make_chart(iris_data_1)
make_chart(iris_data_2)
make_chart(iris_data_3)
make_chart(iris_data_4)
make_chart(iris_data_5)

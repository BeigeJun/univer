import numpy as np
from sklearn.datasets import load_iris
import pandas as pd
Iris = load_iris()
Iris_Data = pd.DataFrame(data=Iris['data'], columns=Iris['feature_names'])
iris_data_1 = np.array(Iris_Data.iloc[:29])
def make(A,B):
    sum = np.zeros(shape=29)
    for i in range(29):
        sum[i] = A[i][B]
    return sum
num = 29

X = make(iris_data_1,0)
Y = make(iris_data_1,1)
from matplotlib.patches import Polygon
from pylab import plt,mpl
plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'
X_mean = X.mean()
Y_mean = Y.mean()
Plus_all_up = 0
Plus_all_down = 0
for i in range(num):
    Plus_all_up += (int(X[i])-int(X_mean))*(int(Y[i])-int(Y_mean))
    Plus_all_down += (X[i]-X_mean)**2
Inc = Plus_all_up/Plus_all_down
B_julpeun = Y.mean() - Inc * X.mean()
print(B_julpeun)
print(Inc)
fig, ax = plt.subplots(figsize=(5,5))
plt.scatter(X,Y)
y = Inc*X+B_julpeun
plt.plot(X, y, label=f'y = {Inc}x + {B_julpeun}', color='blue')
plt.show()
a = float(input("입력 : "))
y = Inc*a+B_julpeun
print(y)

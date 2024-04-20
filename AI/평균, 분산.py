from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np



# iris 데이터셋 불러오기
iris = load_iris()
X = iris.data[:, 2:]  # petal width와 length만 선택
y = iris.target


A_X = X[:120]
B_X = X[120:]
A_y = y[:120]
B_y = y[120:]


# 데이터 출력
print("A 데이터:")
for i in range(len(A_X)):
    print(f"Petal width: {A_X[i][0]}, Petal length: {A_X[i][1]}")

print("\nB 데이터:")
for i in range(len(B_X)):
    print(f"Petal width: {B_X[i][0]}, Petal length: {B_X[i][1]}")
    
class_indices = [np.where(y == i)[0] for i in range(3)]   

for i, indices in enumerate(class_indices):
    class_data = X[indices]
    class_mean = np.mean(class_data, axis=0)
    class_var = np.var(class_data, axis=0)
    # 소수점 2자리까지 반올림
    class_mean_rounded = np.round(class_mean, 2)
    class_var_rounded = np.round(class_var, 2)
    print(f"Class {i}:")
    print(f"  Mean: {class_mean_rounded}")
    print(f"  Variance: {class_var_rounded}")
    
X_normalized = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
print("정규화")
for row in X_normalized:
    print([round(value, 1) for value in row])

#print(X_normalized)

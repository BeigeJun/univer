import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

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

# Iris 데이터셋 로드
iris = load_iris()
X = iris.data
y = iris.target
data_0 = iris.data[0]
data_51 = iris.data[51]
data_101 = iris.data[101]

# 3-means
initial_centroids = np.array([
     ],
    [6.0, 2.5, 5.0, 1.8],
    [7.0, 3.5, 6.0, 2.2]
])

# 사용자가 지정한 초기 중심점을 사용하여 K-Means 알고리즘을 초기화
kmeans_3_manual_init = KMeans(n_clusters=3, init=initial_centroids, n_init=1)
kmeans_3_manual_init.fit(X)
centroids_3 = kmeans_3.cluster_centers_
labels_3 = kmeans_3.labels_
# 시각화
fig, ax = plt.subplots(2, 2, figsize=(12, 12))

# 3-means 시각화
for i in range(3):
    ax[0, 0].scatter(X[y == i, 0], X[y == i, 1], label=f'Iris {i}', c=f'C{i}')
ax[0, 0].set_title('Actual Iris Species')
ax[0, 0].set_xlabel('Sepal Length')
ax[0, 0].set_ylabel('Sepal Width')
ax[0, 0].legend()

for i in range(3):
    ax[0, 1].scatter(X[labels_3 == i, 0], X[labels_3 == i, 1], label=f'Cluster {i}', c=f'C{i}', marker='x')
ax[0, 1].scatter(centroids_3[:, 0], centroids_3[:, 1], marker='o', s=200, c='red', label='Centroids')
ax[0, 1].set_title('3-Means Clustering')
ax[0, 1].set_xlabel('Sepal Length')
ax[0, 1].set_ylabel('Sepal Width')
ax[0, 1].legend()
print("--------------------------")
make_chart(iris_data_A)
make_chart(iris_data_B)
make_chart(iris_data_C)

print("Centroids for 3-Means:")
for i, centroid in enumerate(centroids_3):
    rounded_centroid = [round(coord, 2) for coord in centroid]
    print(f"Cluster {i}: {rounded_centroid}")


def predict_cluster(data, centroids):
    distances = np.sqrt(np.sum((data - centroids[:, np.newaxis])**2, axis=2))
    cluster_indices = np.argmin(distances, axis=0)
    return cluster_indices

# 각 데이터가 속하는 클러스터 예측
predicted_clusters_A = predict_cluster(iris_data_A, centroids_3)
predicted_clusters_B = predict_cluster(iris_data_B, centroids_3)
predicted_clusters_C = predict_cluster(iris_data_C, centroids_3)

# 각 데이터의 클러스터 출력
print("Predicted Clusters for Iris Data Set A:")
print(predicted_clusters_A)
print("Predicted Clusters for Iris Data Set B:")
print(predicted_clusters_B)
print("Predicted Clusters for Iris Data Set C:")
print(predicted_clusters_C)

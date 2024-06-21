import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 텍스트 파일에서 데이터 읽기
# 예제 파일 경로: 'iris.data.txt'
file_path = 'iris.data.txt'
# Iris 데이터셋에는 헤더가 없으므로, 열 이름을 직접 지정합니다.
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
iris_data = pd.read_csv(file_path, header=None, names=column_names)

# 특성 데이터 추출 (class 열 제외)
features = iris_data.iloc[:, :-1]

# 데이터 표준화
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# K-means 클러스터링 수행 (클러스터 수: 3)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(scaled_features)
clusters = kmeans.labels_

# PCA를 사용하여 2D 시각화
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_features)

# 데이터 시각화
plt.figure(figsize=(10, 7))
plt.scatter(principal_components[:, 0], principal_components[:, 1], c=clusters, cmap='viridis', marker='o', edgecolor='k', s=100)
plt.title('K-means Clustering of Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster Label')
plt.show()

# 클러스터 중심 출력
print("Cluster Centers (in standardized scale):")
print(kmeans.cluster_centers_)

# 클러스터 레이블을 원본 데이터프레임에 추가
iris_data['cluster'] = clusters
print(iris_data.head())

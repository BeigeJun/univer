from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# IRIS 데이터셋 로드
iris = load_iris()
X, y = iris.data, iris.target

# 처음 10개의 샘플만 사용하여 모델 학습
X_train, X_test, y_train, y_test = train_test_split(X[:30], y[:30], test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# 나머지 120개의 샘플로 분류 작업 수행
X_unseen, y_unseen = X[30:], y[30:]
y_pred = knn.predict(X_unseen)

# 인식률 계산
accuracy = accuracy_score(y_unseen, y_pred)
print("인식률:", accuracy)

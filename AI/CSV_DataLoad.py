import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt

# device 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.cuda.manual_seed_all(777)
print(device + " is using")

# CSV 파일 로드
train_df = pd.read_csv('/content/sample_data/mnist_train.csv', header=None)
test_df = pd.read_csv('/content/sample_data/mnist_test.csv', header=None)

# 데이터 및 레이블 분리
y_train = train_df.iloc[:, 0].values
X_train = train_df.iloc[:, 1:].values

y_test = test_df.iloc[:, 0].values
X_test = test_df.iloc[:, 1:].values

# 데이터 정규화 (0-255 사이의 픽셀 값을 0-1 사이로 변환)
X_train = X_train / 255.0
X_test = X_test / 255.0

# 데이터 텐서로 변환
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).view(-1, 1, 28, 28)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32).view(-1, 1, 28, 28)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 텐서 데이터셋 생성
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# 데이터로더 생성
trainLoader = DataLoader(train_dataset, batch_size=100, shuffle=True)
testLoader = DataLoader(test_dataset, batch_size=100, shuffle=False)

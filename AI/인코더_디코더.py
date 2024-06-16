import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# AutoencoderMLP 클래스 정의
class AutoencoderMLP(nn.Module):
    def __init__(self):
        super(AutoencoderMLP, self).__init__()
        # 인코더 부분
        self.encoder_fc1 = nn.Linear(10, 32)
        self.encoder_fc2 = nn.Linear(32, 16)
        self.encoder_fc3 = nn.Linear(16, 3)  # 잠재 공간

        # 디코더 부분
        self.decoder_fc1 = nn.Linear(3, 16)
        self.decoder_fc2 = nn.Linear(16, 32)
        self.decoder_fc3 = nn.Linear(32, 10)

    def forward(self, x):
        # 인코더
        x = F.relu(self.encoder_fc1(x))
        x = F.relu(self.encoder_fc2(x))
        x = F.relu(self.encoder_fc3(x))

        # 디코더
        x = F.relu(self.decoder_fc1(x))
        x = F.relu(self.decoder_fc2(x))
        x = torch.sigmoid(self.decoder_fc3(x))
        return x

# 모델 인스턴스화
model = AutoencoderMLP()

# 데이터셋 준비
inputs = torch.rand(100, 10)  # 100개의 10차원 벡터
targets = inputs  # Autoencoder의 목표는 입력을 복원하는 것

# 데이터로더 (배치 크기: 10)
dataset = TensorDataset(inputs, targets)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# 손실 함수 및 옵티마이저 정의 (SGD 사용)
criterion = nn.MSELoss()  # Mean Squared Error 손실 함수
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 학습 루프
num_epochs = 100  # 에폭 수

for epoch in range(num_epochs):
    for batch_inputs, batch_targets in dataloader:
        # 모델의 예측값 계산
        outputs = model(batch_inputs)

        # 손실 계산
        loss = criterion(outputs, batch_targets)

        # 역전파 및 옵티마이저 단계
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 학습이 끝난 후, 모델을 평가
model.eval()  # 평가 모드로 전환

# 새로운 입력 데이터 생성 (10차원 벡터)
test_inputs = torch.rand(10, 10)  # 10개의 10차원 벡터

# 모델을 통해 테스트 입력을 처리
with torch.no_grad():  # 평가 시에는 기울기 계산을 하지 않음
    reconstructed_outputs = model(test_inputs)

# 출력 결과 확인
print("테스트 입력:")
print(test_inputs)
print("재구성된 출력:")
print(reconstructed_outputs)

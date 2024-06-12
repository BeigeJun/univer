import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torchvision import datasets, transforms
#입력, 출력값 선언
Input = torch.FloatTensor([[1.0, 1.0, 1.0, 1.0, 1.0,
                           0.1, 0.1, 1.0, 0.1, 0.1,
                           0.1, 0.1, 1.0, 0.1, 0.1,  # T
                           0.1, 0.1, 1.0, 0.1, 0.1,
                           0.1, 0.1, 1.0, 0.1, 0.1],
                           [0.9, 0.9, 0.9, 0.9, 0.9,
                            0.9, 0.1, 0.1, 0.1, 0.1,
                            0.9, 0.1, 0.1, 0.1, 0.1,  # C
                            0.9, 0.1, 0.1, 0.1, 0.1,
                            0.9, 0.9, 0.9, 0.9, 0.9],
                          [1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 0.1, 0.1, 0.1, 0.1,
                           1.0, 1.0, 1.0, 1.0, 1.0,  # E
                           1.0, 0.1, 0.1, 0.1, 0.1,
                           1.0, 1.0, 1.0, 1.0, 1.0]])
Output = torch.FloatTensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
#신경망 모델 만들기
class MNISTNet(torch.nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.fc1 = torch.nn.Linear(5*5, 20)
        self.fc2 = torch.nn.Linear(20, 10)
        self.fc3 = torch.nn.Linear(10, 3)

    def forward(self, x):
        x = x.view(-1, 5 * 5) #이미지 평탄화
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x

Model = MNISTNet()

criterion = nn.MSELoss() #BCELoss란 PyTorch에서 제공하는 이진 분류(binary classification)를 위한 손실 함수(loss function) 중 하나인 이진 교차 엔트로피 손실(Binary Cross Entropy Loss)
optimizer = torch.optim.SGD(Model.parameters(), lr=0.01)
#모델의 파라미터를 SGD 옵티마이저를 사용하여 업데이트하기 위한 설정을 생성
#Model.parameters(): 모델의 파라미터들을 인자로 전달
#lr : 학습률(learning rate)
for epoch in range(10001):
    optimizer.zero_grad() #그래디언트 초기화

    reselt = Model(Input) #순전파

    Error = criterion(reselt, Output) #Error값 계산후 역전파 진행

    Error.backward() #역전파 실행 그래디언트(기울기 추출)
    optimizer.step() #역전파를 통해 얻은 기울기로 가중치 업데이트
    #위 두개는 세트

    if epoch % 100 == 0:
        print(epoch, Error.item())
    if Error < 0.03:
      break
Input = torch.FloatTensor(
                [[0.9, 0.9, 0.9, 0.9, 0.9,
               0.1, 0.1, 0.9, 0.1, 0.1,
               0.1, 0.1, 0.9, 0.1, 0.1,  # T
               0.1, 0.1, 0.9, 0.1, 0.1,
               0.1, 0.1, 0.9, 0.1, 0.1],
              [0.9, 0.9, 0.9, 0.9, 0.9,
               0.9, 0.1, 0.1, 0.1, 0.1,
               0.9, 0.1, 0.1, 0.1, 0.1,  # C
               0.9, 0.1, 0.1, 0.1, 0.1,
               0.9, 0.9, 0.9, 0.9, 0.1],
              [0.9, 0.9, 0.9, 0.9, 0.1,
               0.9, 0.1, 0.1, 0.1, 0.1,
               0.9, 0.9, 0.9, 0.9, 0.9,  # E
               0.9, 0.1, 0.1, 0.1, 0.1,
               0.9, 0.9, 0.9, 0.9, 0.1],
                [0.1, 0.9, 0.9, 0.9, 0.9,
                 0.9, 0.1, 0.1, 0.1, 0.1,
                 0.9, 0.1, 0.9, 0.9, 0.9,  # G
                 0.9, 0.1, 0.1, 0.1, 0.9,
                 0.1, 0.9, 0.9, 0.9, 0.1],
                [0.9, 0.9, 0.9, 0.9, 0.1,
                 0.9, 0.1, 0.1, 0.1, 0.9,
                 0.9, 0.9, 0.9, 0.9, 0.9,  # B
                 0.9, 0.1, 0.1, 0.1, 1.0,
                 0.9, 0.9, 0.9, 0.9, 0.1]])

Reselt = Model(Input)
Max = torch.argmax(Reselt, dim=1) + 1
print('출력값: ', Max.numpy())

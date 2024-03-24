import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_image_root = "C:/Users/wns20/PycharmProjects/얼굴인식/face_data"
test_image_root = "C:/Users/wns20/PycharmProjects/얼굴인식/face_test"
train_loader = ImageFolder(root=train_image_root, transform=transform)
test_loader = ImageFolder(root=test_image_root, transform=transform)
print("데이터셋 크기:", len(train_loader))
print("클래스:", train_loader.classes)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=9, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=9, out_channels=18, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=18, out_channels=27, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=27, out_channels=36, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(36, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.fc1(x.view(-1, 36)))
        x = self.fc2(x)
        return x

cnn = CNN()
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn.parameters(), lr=0.01)

cnn.train()
for epoch in range(1000):
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs = inputs.unsqueeze(1)  # 배치 차원을 1로 추가
        optimizer.zero_grad()

        outputs = cnn(inputs)

        loss = criterion(outputs, torch.LongTensor([labels]))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    if epoch % 100 == 99:
        print('[%d] loss: %.5f' % (epoch + 1, running_loss / 100))
        running_loss = 0.0


cnn.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data = data.unsqueeze(1)
        output = cnn(data)
        test_loss += criterion(output, torch.tensor([target])).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(torch.LongTensor([target])).sum().item()
print(correct)

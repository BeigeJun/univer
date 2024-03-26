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
train_dataset = ImageFolder(root=train_image_root, transform=transform)
test_loader = ImageFolder(root=test_image_root, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=5, shuffle=True)
# print("데이터셋 크기:", len(train_loader))
# print("클래스:", train_loader.classes)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=9, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=9, out_channels=18, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=18, out_channels=27, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=27, out_channels=36, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(36, 100)
        self.fc2 = nn.Linear(100, 2)

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
        x = F.softmax(x, dim=1)
        return x

cnn = CNN()
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn.parameters(), lr=0.01)

cnn.train()
for epoch in range(10000):
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        optimizer.zero_grad()

        outputs = cnn(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    if epoch % 100 == 99:
        print('[%d] loss: %.10f' % (epoch + 1, running_loss / 100))
        running_loss = 0.0


cnn.eval()
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data = data.unsqueeze(1)
        output = cnn(data)
        pred = output.argmax(dim=1, keepdim=True)

        print("예측률 :", output[0][0], ",", output[0][1])
        correct += pred.eq(torch.LongTensor([target])).sum().item()
print(correct)

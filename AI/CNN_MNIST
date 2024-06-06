import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn.init
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=9, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=9, out_channels=18, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(5 * 5 * 18, 100)
        self.fc2 = nn.Linear(100, 10)

        torch.nn.init.uniform_(self.fc1.weight, a=0.01, b=0.2)
        torch.nn.init.uniform_(self.fc2.weight, a=0.01, b=0.2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = x.view(-1, 5 * 5 * 18)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

cnn = CNN()
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn.parameters(), lr=0.01)

cnn.train()
for epoch in range(100):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()

        outputs = cnn(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print('[%d, %d] loss: %.5f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
        if running_loss < 0.03:
          break


cnn.eval()  # test case 학습 방지를 위함
test_loss = 0
correct = 0
with torch.no_grad():

    for data, target, i in test_loader:
       output = cnn(data)
       test_loss += criterion(output, target).item()
       pred = output.argmax(dim=1, keepdim=True)
       correct += pred.eq(target.view_as(pred)).sum().item()
       if i == 2:
        break
print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
transform = transforms.Compose([
    transforms.ToTensor(),

])
train_dataset = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=9, kernel_size=3, stride=1, padding=1) #인체널 1(그레이스케일),


cnn = CNN()


cnn.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    print(test_loader)
    print(images[0])
    First_pool_outputs = F.max_pool2d(images, kernel_size=2, stride=2)

    for i in range(1):
        plt.subplot(1, 3, i + 1)
        plt.imshow(First_pool_outputs[0][i].squeeze().detach().numpy(), cmap='gray')
        print(First_pool_outputs[0][i]);
        print("---------------------------------")
        plt.title("First Pooling Output")
    plt.show()
    data_iter = iter(test_loader)
    images, labels = next(data_iter)

    First_pool_outputs = F.max_pool2d(images, kernel_size=2, stride=2)

    for i in range(1):
        plt.subplot(1, 3, i + 1)
        plt.imshow(First_pool_outputs[1][i].squeeze().detach().numpy(), cmap='gray')

        print(First_pool_outputs[1][i]);
        print("---------------------------------")
        plt.title("First Pooling Output")
    plt.show()
    data_iter = iter(test_loader)
    images, labels = next(data_iter)

    First_pool_outputs = F.max_pool2d(images, kernel_size=2, stride=2)

    for i in range(1):
        plt.subplot(1, 3, i + 1)
        plt.imshow(First_pool_outputs[2][i].squeeze().detach().numpy(), cmap='gray')
        print(First_pool_outputs[2][i]);
        plt.title("First Pooling Output")
    plt.show()

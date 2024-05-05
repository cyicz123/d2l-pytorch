import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_data_fashion_mnist(batch_size=256):
    trans = transforms.ToTensor()

    mnist_train = datasets.FashionMNIST(root="../data", train=True, download=True, transform=trans)
    mnist_test = datasets.FashionMNIST(root="../data", train=False, download=True, transform=trans)

    train_iter = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

    return train_iter, test_iter

net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10)
)

batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)


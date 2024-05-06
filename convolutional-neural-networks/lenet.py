import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_data_fashion_mnist(batch_size=256):
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    mnist_train = datasets.FashionMNIST(root="../data", train=True, download=True, transform=trans)
    mnist_test = datasets.FashionMNIST(root="../data", train=False, download=True, transform=trans)

    train_iter = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=4)
    test_iter = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_iter, test_iter

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)

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
lr = 0.4
net.apply(init_weights)
optimizer = torch.optim.SGD(net.parameters(), lr = lr)
loss = nn.CrossEntropyLoss()
num_epochs = 10

batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)

for epoch in range(num_epochs):
    net.train()
    running_loss = 0.0
    for i, (x, y) in enumerate(train_iter):
        optimizer.zero_grad()
        y_hat = net(x)
        l = loss(y_hat, y)
        l.backward()
        optimizer.step()

        running_loss += l.item()
    
    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_iter)}")

print("Finished Training")

# 测试模型
correct = 0
total = 0
with torch.no_grad():
    for (x, y) in test_iter:
        y_hat = net(x)  # 前向传播
        _, predicted = torch.max(y_hat.data, 1)  # 找到最大概率的类别
        total += y.size(0)
        correct += (predicted == y).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy on the test set: {accuracy}%")
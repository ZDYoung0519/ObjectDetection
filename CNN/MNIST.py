import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 输入1通道，输出10通道，kernel 5*5
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.conv3 = nn.Conv2d(20, 40, 3)

        self.mp = nn.MaxPool2d(2)
        # fully connect
        self.fc = nn.Linear(40, 10)  # （in_features, out_features）

    def forward(self, x):
        # 输入x的维度是(batch_size,channels,w,h),
        # x: 64*1*28*28
        x = F.relu(self.mp(self.conv1(x)))
        # x: 64*10*12*12  feature map =[(28-4)/2]^2=12*12
        x = F.relu(self.mp(self.conv2(x)))
        # x: 64*20*4*4
        x = F.relu(self.mp(self.conv3(x)))
        x = x.view(x.size(0), -1)
        # print(x.size())
        # x: 64*320
        x = self.fc(x)
        # x:64*10
        # print(x.size())
        return x  # 64*10


# 参数设置
batch_size = 32
epoch_num = 10
LR = 0.01
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MNIST数据集下载,download=False表示已经下载过,未下载自己改为True
train_dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor(), download=False)
test_dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 定义损失函数和优化方式
model = Net()
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.5)
criterion = nn.CrossEntropyLoss()

# 训练
model.train()
for epoch in range(epoch_num):
    for idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()  # 所有参数的梯度清零
        loss.backward()  # 即反向传播求梯度
        optimizer.step()  # 调用optimizer进行梯度下降更新参数
        if idx % 100 == 0:
            print('Epoch{:d}, batch:{:d}, loss:{:.2f}.'.format(epoch, idx, loss))

# 预测
model.eval()
test_loss = 0
ACC = 0
num = 0
with torch.no_grad():
    for idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += criterion(output, target)
        num += len(target)
        output_class = output.argmax(dim=1)
        ACC += (output_class == target).sum().item()
    print('Test loss:{:.2f}, accuracy:{:.2f}%'.format(test_loss, ACC/num*100))

import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
# 创建数据集
train_data = torchvision.datasets.MNIST('../MNIST_data', train=True, download=True,
                                        transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.MNIST('../MNIST_data', train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())
# Loader加载数据集
train_loader = DataLoader(train_data, batch_size=64, shuffle=True, drop_last=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True, drop_last=True)
train_size = len(train_data)
test_size = len(test_data)

# 搭建网络模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1,10,kernel_size=5)
        self.conv2 = nn.Conv2d(10,20,kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320,50)
        self.fc2 = nn.Linear(50,10)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x),kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)),kernel_size=2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


# 创建网络模型
model = CNN()
# 设置Loss
loss_fn = nn.CrossEntropyLoss()
# 设置优化器
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
# 训练参数设置
epoch = 10
total_train_step = 0
total_test_step = 0

# 日志记录
writer = SummaryWriter('../logs/MNIST_CNN')

for i in range(epoch):
    #训练开始
    model.train(True)
    print("---第{}轮训练---".format(i+1))
    for data in train_loader:
        # 训练数据获取
        images, labels = data
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        # 优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 数据日志记录
        total_train_step += 1
        writer.add_scalar('train_loss', loss.item(), total_train_step)
        if(total_train_step % 100 == 0):
            print("第{}次训练，Loss:{}".format(total_train_step,loss))

    # 对本轮训练评估
    model.eval()
    print("第{}轮评估".format(i+1))
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            #获取测试数据
            images, labels = data
            outputs = model(images)
            test_loss = loss_fn(outputs, labels)
            # 数据日志记录
            total_train_step+=1
            total_test_loss+= test_loss.item()
            accuracy = (outputs.argmax(dim=1) == labels).sum()
            total_accuracy+=accuracy
    writer.add_scalar('test_loss', total_test_loss,i)
    writer.add_scalar('test_accuracy', total_accuracy/test_size,i)
    print("测试Loss：{},准确度：{}".format(total_test_loss,total_accuracy/test_size))
    # 模型保存
    torch.save(model,'../model/MNIST_CNN/model{}.pth'.format(i+1))
    print("模型{}已存档".format(i+1))
writer.close()


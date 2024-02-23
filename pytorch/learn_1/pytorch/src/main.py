import torch
from torch.utils.data import DataLoader
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter

# 创建数据集
train_data = torchvision.datasets.CIFAR10('../data', train=True, download=True,
                                          transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10('../data', train=False, download=True,
                                         transform=torchvision.transforms.ToTensor())

# DataLoad加载数据集
train_loader = DataLoader(dataset=train_data, batch_size=64,drop_last=True)
test_loader = DataLoader(dataset=test_data, batch_size=64,drop_last=True)
train_size = len(train_data)
test_size = len(test_data)

# 搭建神经网络
class NN_Model(nn.Module):
    def __init__(self):
        super(NN_Model, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=5,stride=1,padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=5,stride=1,padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,stride=1,padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(64*4*4,64),
            nn.Linear(64,10)
        )
    def forward(self, x):
        x = self.model(x)
        return x

# 创建网络模型
model = NN_Model()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
optim = torch.optim.SGD(model.parameters(), lr=0.01)

# 设置训练参数
total_train_step = 0
total_test_step = 0

# 日志记录
writer = SummaryWriter(log_dir='../logs')
#训练轮数
epoch = 10
for i in range(epoch):
    print("----这是第{}轮训练----".format(i+1))
    # 训练开始
    model.train()
    for data in train_loader:
        # 预测数据
        imgs, labels = data
        outputs = model(imgs)
        loss = loss_fn(outputs, labels)
        # 优化模型
        optim.zero_grad()
        loss.backward()
        optim.step()
        # 数据日志记录
        total_train_step +=1
        writer.add_scalar('train_loss', loss.item(), total_train_step)
        if(total_train_step % 100 == 0):
            print("第{}次训练,loss:{}".format(total_train_step, loss))

    # 测试开始
    model.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            # 测试数据
            imgs, labels = data
            outputs = model(imgs)
            test_loss = loss_fn(outputs,labels)
            # 数据日志记录
            total_test_loss += test_loss
            total_test_step += 1
            accuracy = (outputs.argmax(dim=1) == labels).sum()
            total_accuracy += accuracy
    print("测试集Aver_Loss:{}".format(total_test_loss/total_test_step))
    print("测试集Accuracy:{}".format(total_accuracy/test_size))
    writer.add_scalar('Aver_test_loss', total_test_loss/test_size, total_test_step)
    writer.add_scalar('test_accuracy', total_accuracy/test_size, total_test_step)
    # 模型保存
    torch.save(model.state_dict(),'../model/model{}.pth'.format(i+1))
    print("模型{}已存档".format(i+1))

writer.close()


# Reference: https://github.com/yunjey/pytorch-tutorial/
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_epochs = 80
batch_size = 100
learning_rate = 0.001

transformer = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()
])

# CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='../../../_data/',
                                             train=True, 
                                             transform=transformer,
                                             download=True)

test_dataset = torchvision.datasets.CIFAR10(root='../../../_data/',
                                            train=False, 
                                            transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # conv layer의 결과로 차원이 축소된다면 원본 이미지의 차원도 축소 해야한다.
        # 그 지점이 바로 downsample이다.
        if self.downsample:
            residual = self.downsample(x)

        # Residual Block에서 F(x) + x 부분
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    # block = Residual Block (위에서 정의한 ResBlock 클래스이다.)
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layers1 = self.make_layer(block, 16, layers[0])
        self.layers2 = self.make_layer(block, 32, layers[1], 2)
        self.layers3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

        # (W - F + 2P)/S + 1
        # W - F / S + 1
    
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels) :
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layers1(out)
        out = self.layers2(out)
        out = self.layers3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1) # flatten
        out = self.fc(out)
        return out

model = ResNet(ResBlock, [2,2,2]).to(device)

# Loss
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 학습 도중 Learning Rate을 변경하기 위해 사용한다.
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

total_step = len(train_loader)
curr_lr = learning_rate

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print("Epoch [{}/{}], Step: [{}/{}], Loss:[{}]".format(epoch, num_epochs, i+1, total_step, loss.item()))

    if (epoch+1) % 20 == 0:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)

# Turn to evaluation mode
mode.eval()
with torch.no_grad():
    correct = 0
    total = 0

    for images, labels in test_loader:
        output = model(images)
        _, predicted = torch.max(output, 1)
        total += labels.size(0)
        correct = (predicted == labels).sum().item()
    
    print("Accuracy: [{}]".format(100 * correct / total))

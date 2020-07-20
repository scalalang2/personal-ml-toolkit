# Reference : https://github.com/yunjey/pytorch-tutorial/
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print("We're able to use cuda device")

num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.0001

# 학습 데이터 셋을 저장한다.
train_dataset = torchvision.datasets.MNIST(root='../../../_data',
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

# 테스트 데이터 셋을 저장한다.
test_dataset = torchvision.datasets.MNIST(root='../../../_data',
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data Loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

# CNN 네트워크 정의
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # W = 32, F=5, P=2, S=1
        # input data : 28 x 28 x 1

        # Layer1
        # (W−F+2P)/S + 1 = 28
        # (W - F)/S + 1 = 14
        
        # Layer2 
        # (W−F+2P)/S + 1 = 28
        # (16 - 2)/2 + 1 = 7
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)

        # x.size(0) => batch size를 나타내므로 (batch_size, 7*7*32)가 될 것이다.
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

model = ConvNet(num_classes).to(device)

# 학습기 개발
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 학습 시작
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        # 미분을 모두 계산하고 최적화를 시작해라
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 200 == 0:
            print('Epoch [{}/{}], Step: [{}/{}], Loss: [{}]'.format(epoch, num_epochs, i+1, total_step, loss.item()))

# 모델의 성능을 평가하기
# model.eval() : 학습 단계와 검증 단계에서 동작이 다른 레이어가 있는데 이를 on/off 하기 위해 사용하는 함수이다.
# 예를 들어 Dropout 레이어 같은 경우에는 학습 단계에서는 뉴런을 Drop시키고 테스트 단계에서는 모든 뉴런을 활성화 해야 한다.
model.eval()
with torch.no_grad():
    correct = 0
    total = 0

    for images, labeles in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        # torch.max(input, dim) returns (values, indexes)
        _, predicted = torch.max(outputs.data, 1) 
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    print("Test Accuracy: [{}]".format(100 * correct/total))
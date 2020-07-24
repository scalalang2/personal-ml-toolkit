# Conditional GAN
import torch
import torch.nn as nn
import torchvision 
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 하이퍼 파라미터
batch_size = 100
latent_dim = 100
num_classes = 10
num_epochs = 20
learning_rate = 0.0002

transformer = transforms.Compose([
    transforms.ToTensor()
])

train_data = torchvision.datasets.MNIST(
    "../../../_data",
    train=True,
    transform=transformer,
    download=True
)

test_data = torchvision.datasets.MNIST(
    "../../../_data",
    train=False,
    transform=transformer
)

train_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

# 모델 정의 : 판별기
class CGAN_D(nn.Module):
    def __init__(self):
        super(CGAN_D, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(784 + num_classes, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x, cond):
        one_hot = torch.FloatTensor(cond.size(0), num_classes).zero_()

        # scatter 함수란?
        one_hot = torch.scatter(one_hot, 1, cond.view(-1, 1), 1)
        out = torch.cat([x.view(x.size(0), -1), one_hot], 1)
        out = self.model(out)
        return out

# 모델 정의 : 생성기
class CGAN_G(nn.Module):
    def __init__(self):
        super(CGAN_G, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128),
            nn.BatchNorm1d(128, 0.8),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256, 0.8),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512, 0.8),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, 0.8),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, z, cond):
        one_hot = torch.FloatTensor(cond.size(0), num_classes).zero_()
        one_hot = torch.scatter(one_hot, 1, cond.view(-1, 1), 1)
        out = torch.cat([z, one_hot], 1)
        out = self.model(out)
        return out

# 모델 생성
Model_D = CGAN_D()
Model_G = CGAN_G()

criterion = nn.BCELoss()
optimizer_D = torch.optim.Adam(Model_D.parameters(), lr=learning_rate)
optimizer_G = torch.optim.Adam(Model_G.parameters(), lr=learning_rate)

total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        z = torch.Tensor(np.random.normal(0, 1, (images.size(0), latent_dim)))
        z_labels = torch.LongTensor(np.random.randint(0, num_classes, batch_size))

        valid = torch.FloatTensor(images.size(0), 1).fill_(1.0)
        fake = torch.FloatTensor(images.size(0), 1).fill_(0.0)

        # generator loss
        optimizer_G.zero_grad()
        real_images = images
        fake_images = Model_G(z, z_labels)
        
        output = Model_D(fake_images, z_labels)
        g_loss = criterion(output, valid)
        
        g_loss.backward()
        optimizer_G.step()

        # discriminator loss
        optimizer_D.zero_grad()
        real_output = Model_D(real_images, labels)
        fake_output = Model_D(fake_images.detach(), z_labels)
        real_loss = criterion(real_output, valid)
        fake_loss = criterion(fake_output, fake)

        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        if (i+1) % 50 == 0:
            print("Epoch [{}/{}], Step [{}/{}], G loss [{}], D loss [{}]".format(epoch, num_epochs, i+1, total_step, g_loss.item(), d_loss.item()))

# Generate Fake Images
num_tests = 10

with torch.no_grad():
    for i in range(1, 9):
        z = torch.Tensor(np.random.normal(0, 1, (num_tests, latent_dim)))
        z_labels = torch.LongTensor(np.random.randint(0, num_classes, batch_size))

        fake_image = Model_G(z, z_labels)
        images = fake_image.reshape(-1, 28, 28)

        plt.imshow(images[0], cmap='gray')
        plt.show()
    
# reference - https://github.com/L1aoXingyu/pytorch-beginner
import os

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

if not os.path.exists('./result'):
    os.mkdir('./result')

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

NUM_EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 0.0001

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) # -- ??
])

print("[1] - Download mnist data.")
dataset = MNIST('~/ML-dataset', transform=img_transform, download=True)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print("[2] - Dataset loaded")
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 16),
            nn.ReLU(True),
            nn.Linear(16, 10)
        )

        self.decoder = nn.Sequential(
            nn.Linear(10, 16),
            nn.ReLU(True),
            nn.Linear(16, 32),
            nn.ReLU(True),
            nn.Linear(32, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 28 * 28)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

print("[3] - initialize model")
model = autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5
)

print("[4] - start learning")
for epoch in range(NUM_EPOCHS):
    for data in dataloader:
        img, _ = data
        img = img.view(img.size(0), -1)
        img = Variable(img)

        output = model(img)
        loss = criterion(output, img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print("epoch [{}/{}], loss [{:.4f}]".format(epoch + 1, NUM_EPOCHS, loss.item()))

    if epoch % 10 == 0:
        pic = to_img(output.data)
        save_image(pic, "./result/result_{}.png".format(epoch))
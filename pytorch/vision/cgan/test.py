# reference : https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cgan/cgan.py
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("./images", exist_ok=True)

num_epochs = 200
batch_size = 64
lr = 0.0002
b1 = 0.5
b2 = 0.999
latent_dim = 100
n_classes = 10
img_size = 32
channels = 1
sample_interval = 400

transformer = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../../_data",
        train=True,
        download=True,
        transform=transformer,
    ),
    batch_size=batch_size,
    shuffle=True,
)


img_shape = (channels, img_size, img_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(n_classes, n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.ReLU())
            return layers
        
        self.model = nn.Sequential(
            *block(latent_dim + n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z, labels):
        gen_input = torch.cat((self.label_emb(labels), z), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_emb = nn.Embedding(n_classes, n_classes)

        self.model = nn.Sequential(
            nn.Linear(n_classes + int(np.prod(img_shape)), 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1)
        )

    def forward(self, img, labels):
        d_in = torch.cat((img.view(img.size(0), -1), self.label_emb(labels)), -1)
        validity = self.model(d_in)
        return validity

criterion = torch.nn.MSELoss()
generator = Generator().to(device)
discriminator = Discriminator().to(device)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr)

def sample_image(n_row, batches_done):
    z = torch.FloatTensor(np.random.normal(0, 1, n_row**2, latent_dim)).to(device)
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = torch.LongTensor(labels).to(device)
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)

# Train the CGAN network
total_batch = len(dataloader)
for epoch in range(num_epochs):
    for i, (imgs, labels) in enumerate(dataloader):
        batch_size = imgs.shape[0]

        valid = torch.FloatTensor(batch_size, 1).fill_(1.0).requires_grad_(False).to(device)
        fake = torch.FloatTensor(batch_size, 1).fill_(0.0).requires_grad_(False).to(device)
        real_images = torch.FloatTensor(imgs).to(device)
        labels = torch.LongTensor(labels).to(device)

        # train generator
        optimizer_G.zero_grad()
        z = torch.FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))).to(device)
        gen_labels = torch.LongTensor(np.random.randint(0, n_classes, batch_size)).to(device)

        output_g = generator(z, gen_labels)
        output_d = discriminator(output_g, gen_labels)
        g_loss = criterion(output_d, valid)
        
        g_loss.backward()
        optimizer_G.step()

        # train discriminator
        optimizer_D.zero_grad()

        real_output = discriminator(real_images, labels)
        real_loss = criterion(real_output, valid)

        fake_output = discriminator(output_g.detach(), gen_labels)
        fake_loss = criterion(fake_output, fake)

        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()

        optimizer_D.step()

        if (i+1) % 200 == 0:
            print("Epoch: [{}/{}] Batch: [{}/{}], D_loss: [{}], G_loss: [{}]".format(epoch, num_epochs, i, total_batch, d_loss, g_loss))
    
    sample_image(n_row=10, batches_done=epoch)

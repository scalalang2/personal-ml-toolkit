# reference - https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy

# set path
CONTENT_DATA_PATH   = "/Users/idohyeon/ML-dataset/neural-transfer/content.jpeg"
STYLE_DATA_PATH     = "/Users/idohyeon/ML-dataset/neural-transfer/style.jpg"

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 512 if torch.cuda.is_available() else 128

# transforms the image with (resize -> tensor)
loader = transforms.Compose([
    transforms.Resize(imsize),
    transforms.ToTensor()
])

# show images
def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

style_image = image_loader(STYLE_DATA_PATH)
content_image = image_loader(CONTENT_DATA_PATH)

print("shape(style_image): ", style_image.shape)
print("shape(content_image): ", content_image.shape)

unloader = transforms.ToPILImage()
# plt.ion()

def imshow(_image, title=None):
    image = _image.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    # plt.pause(0.001)

plt.figure()
imshow(style_image, title="Style Image")

plt.figure()
imshow(content_image, title="Content Image")
plt.show()

# define content loss, X -> X'
class ContentLoss(nn.Module):
    def __init__(super, target):
        super(ContentLoss).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

# gram matrix ?
def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a*b, c * d) # (3, 16384)
    G = torch.mm(features, features.t())

    # normalize ?
    return G.div(a*b*c*d)

# import pre-trained network
# -> VGG19 모델이 Normalization할 때 이용한 값으로 기존 이미지를 Normalize한다.
cnn = models.vgg19(pretrained=True).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# Normalize
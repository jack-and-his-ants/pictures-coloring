import torch
import numpy as np
import joblib
import matplotlib.pyplot as plt
from src.utils.device import get_device
from src.models.colorization_network import ColorizationNetwork

from PIL import Image
from skimage import color
import torch
import torch.nn as nn
from torchvision.models import (
    resnet18,
    ResNet18_Weights
)


class ColorizationResNetUNet(nn.Module):

    def __init__(self, n_classes=32):
        super().__init__()

        backbone = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        for param in backbone.parameters():
            param.requires_grad = False

        for param in backbone.layer3.parameters():
            param.requires_grad = True

        for param in backbone.layer4.parameters():
            param.requires_grad = True

        self.conv1 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu
        )

        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # Decoder

        self.up1 = nn.ConvTranspose2d(
            512, 256,
            kernel_size=2,
            stride=2
        )

        self.dec1 = nn.Sequential(
            nn.Conv2d(512,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Dropout2d(0.1),

            nn.Conv2d(256,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.up2 = nn.ConvTranspose2d(
            256,128,
            kernel_size=2,
            stride=2
        )

        self.dec2 = nn.Sequential(
            nn.Conv2d(256,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Dropout2d(0.1),

            nn.Conv2d(128,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.up3 = nn.ConvTranspose2d(
            128,64,
            kernel_size=2,
            stride=2
        )

        self.dec3 = nn.Sequential(
            nn.Conv2d(128,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Dropout2d(0.1),

            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.up4 = nn.ConvTranspose2d(
            64,64,
            kernel_size=2,
            stride=2
        )

        self.dec4 = nn.Sequential(
            nn.Conv2d(128,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Dropout2d(0.1),

            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.up5 = nn.ConvTranspose2d(
            64,32,
            kernel_size=2,
            stride=2
        )

        self.dec5 = nn.Sequential(
            nn.Conv2d(32,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.head = nn.Conv2d(
            32,
            n_classes,
            kernel_size=1
        )

    def forward(self, x):

        x = x.repeat(1,3,1,1)


        mean = torch.tensor(
            [0.485,0.456,0.406],
            device=x.device
        ).view(1,3,1,1)

        std = torch.tensor(
            [0.229,0.224,0.225],
            device=x.device
        ).view(1,3,1,1)

        x = (x / 100.0 - mean) / std

        x0 = self.conv1(x)

        x1 = self.layer1(
            self.maxpool(x0)
        )

        x2 = self.layer2(x1)

        x3 = self.layer3(x2)                # 8x8

        x4 = self.layer4(x3)                # 4x4

        d1 = self.up1(x4)
        d1 = torch.cat([d1,x3],dim=1)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([d2,x2],dim=1)
        d2 = self.dec2(d2)

        d3 = self.up3(d2)
        d3 = torch.cat([d3,x1],dim=1)
        d3 = self.dec3(d3)

        d4 = self.up4(d3)
        d4 = torch.cat([d4,x0],dim=1)
        d4 = self.dec4(d4)

        d5 = self.up5(d4)
        d5 = self.dec5(d5)

        return self.head(d5)

def colorize_image(
    image_path,
    model,
    kmeans,
    image_size=(128, 128),
    device="mps"
):


    rgb = np.array(
        Image.open(image_path)
        .convert("RGB")
        .resize(image_size)
    )



    lab = color.rgb2lab(rgb / 255.0)

    l_channel = lab[:, :, 0]


    x = torch.tensor(
        l_channel,
        dtype=torch.float32
    )

    x = x.unsqueeze(0).unsqueeze(0)
    x = x.to(device)



    model.eval()

    with torch.no_grad():

        logits = model(x)

        pred_classes = logits.argmax(dim=1)

    pred_classes = pred_classes.squeeze().cpu().numpy()



    ab = kmeans.cluster_centers_[pred_classes]



    reconstructed_lab = np.zeros(
        (
            l_channel.shape[0],
            l_channel.shape[1],
            3
        ),
        dtype=np.float32
    )

    reconstructed_lab[:, :, 0] = l_channel
    reconstructed_lab[:, :, 1:] = ab


    reconstructed_rgb = color.lab2rgb(
        reconstructed_lab
    )

    return (
        rgb,
        reconstructed_rgb,
        pred_classes
    )

device = get_device()

model = ColorizationResNetUNet(
    n_classes=32
)

checkpoint = torch.load(
    "/Users/jacek/Desktop/GitHubProjects/pictures-coloring/notebooks/checkpoints/best_model.pth",
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)

kmeans = joblib.load(
    "/Users/jacek/Desktop/GitHubProjects/pictures-coloring//test_kmeans.pkl"
)

original, colorized, classes = colorize_image(
    "/Users/jacek/Desktop/GitHubProjects/pictures-coloring/data/test/Image_15.jpg",
    model,
    kmeans,
    device=device
)

plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.imshow(original)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(
    color.rgb2gray(original),
    cmap="gray"
)
plt.title("Input L")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(colorized)
plt.title("Colorized")
plt.axis("off")

plt.show()


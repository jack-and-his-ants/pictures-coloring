import torch
import torch.nn as nn
from torchvision.models import (
    resnet18,
    ResNet18_Weights
)


class ColorizationNetwork(nn.Module):

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

        x3 = self.layer3(x2)

        x4 = self.layer4(x3)

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
import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, kernel_size: int = 3, activation: str = "leaky_relu"):

        """
        A conv block with batch normalization and an activation function
        Args
            in_features: 
                The number of input channels
            out_features: 
                The number of output channels
            kernel_size: 
                The kernel size of the convolution block
            activation: 
                The activation function to use
        """

        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size, padding="same"),
            nn.BatchNorm2d(out_features)
        )

        if activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.1)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise f"activation function {activation} unknown"
    
    def forward(self, x):
        return self.activation(self.conv(x))


class Downsample(nn.Module):
    def __init__(self, in_features: int, out_features: int, kernel_size: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_features, out_features, kernel_size)
        )
    
    def forward(self, x: torch.Tensor):
        return self.block(x)


class Upsample(nn.Module):
    def __init__(self, in_features: int, out_features: int, kernel_size: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2),
            ConvBlock(in_features, out_features, kernel_size)
        )
    
    def forward(self, x: torch.Tensor):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__(channels=3)
        self.c1 = ConvBlock(channels, 32)
        self.c2 = ConvBlock(32, 32)
        self.d1 = Downsample(32, 64)
        self.c3 = ConvBlock(64, 64)
        self.c4 = ConvBlock(64, 64)
        self.d2 = Downsample(64, 128)
        self.c5 = ConvBlock(128, 128)
        
        self.u1 = Upsample(128, 64)
        self.c6 = ConvBlock(128, 64)
        self.c7 = ConvBlock(64, 64)
        self.u2 = Upsample(64, 32)
        self.c8 = ConvBlock(64, 32)
        self.c9 = ConvBlock(32, channels, activation="tanh")

    def forward(self, x: torch.Tensor, t: int):
        B, C, H, W = x.shape
        device = x.device

        image_size = C*H*W

        inds = torch.linspace(0, image_size, image_size, device=device).repeat(B, 1).view(-1, C, H, W)

        phase = t/(1000**(2*inds/image_size))
        pe = phase.sin()

        skips = []
        x = self.c2(self.c1(x + pe))
        skips.append(x)
        x = self.c4(self.c3(self.d1(x)))
        skips.append(x)
        x = self.c5(self.d2(x))

        x = self.u1(x)
        x = torch.cat([skips.pop(), x], dim=1)
        x = self.u2(self.c7(self.c6(x)))

        x = torch.cat([skips.pop(), x], dim=1)
        x = self.c9(self.c8(x))
        return x


import torch    
import torch.nn as nn

class FromRGB(nn.Module):
    def __init__(self,out_channels, in_channels=3):
        super().__init__()
        # 1x1 convolution to map 3-channel RGB to feature channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # Activation is required here!
        self.activate = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.activate(x)
        return x
import torch.nn as nn

from .blocks import ConvBlock, DownConv

class Encoder(nn.Module):
    def __init__(self, 
                 in_channels=3, 
                 channels= [32, 64, 128, 256, 512, 1024,2048],
                 norm=nn.BatchNorm2d,
                 activation=nn.ReLU(inplace=True),
                 input_size = (512,512)):
        super().__init__()
        
        self.in_channels = in_channels
        self.channels = channels.copy()
        self.norm = norm
        self.activation= activation
        self.input_size = input_size

        
        
      
        encoder_layers = nn.ModuleList([ConvBlock(self.in_channels, 
                                                  self.channels[0], 
                                                  kernel_size=4, 
                                                  stride=2, 
                                                  padding=1,
                                                  norm=self.norm,
                                                  activation=self.activation)])
        
        for i in range(len(self.channels)-1):
            encoder_layers.append(DownConv(self.channels[i], 
                                           self.channels[i+1],
                                           norm=self.norm,
                                           activation=self.activation))
            
        encoder_layers.append(nn.Flatten())
        self.encoder = nn.Sequential(*encoder_layers)

    
     
    def forward(self, x):
        x = self.encoder(x)
        return x
    

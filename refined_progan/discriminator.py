import torch
import torch.nn as nn
import torch.nn.functional as F


from .blocks import DownsampleBlock

class Discriminator(nn.Module):
    def __init__(self, 
                channels,
                img_channels=3):
        super().__init__()
        self.channels = channels.copy()
        self.channels.reverse()
        self.img_channels = img_channels

        self.initial_block = DownsampleBlock(self.img_channels,
                                             self.channels[0],
                                             groups=None)
        self.blocks = nn.ModuleList()
        for i in range(1,len(self.channels)-1):
            self.blocks.append(
               DownsampleBlock(self.channels[i-1]+self.img_channels,
                               self.channels[i],
                               groups=None)
            )

        self.final_block = nn.Sequential(
                DownsampleBlock(self.channels[-2]+self.img_channels+1,
                                          self.channels[-1],
                                          kernel_size=4,
                                          groups=None),
                nn.Conv2d(self.channels[-1],1,1,1,0)

        )
    
    def minibatch_std(self, x):
        batch_statistics =torch.std(x,dim=0).mean().repeat(x.shape[0],1,x.shape[2],x.shape[3])
        return torch.cat([x,batch_statistics],dim=1)
    
    def forward(self, x):
        y= self.initial_block(x[-1])
        for i in range(len(self.channels)-2):
            y = torch.cat([y,x[-i-2]],dim=1)
            y = self.blocks[i](y)

        y = torch.cat([y,x[0]],dim=1)
        y = self.minibatch_std(y)
        y = self.final_block(y)
        return y.view(y.shape[0],-1)


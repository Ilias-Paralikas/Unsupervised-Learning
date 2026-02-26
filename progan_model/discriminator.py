import torch
import torch.nn as nn

from .blocks import WSConv2d, ConvBlock
class Discriminator(nn.Module):
    def __init__(self,channels, img_channels=3):
        super().__init__()
        self.channels = channels.copy()
        self.img_channels = img_channels
        self.total_steps = len(self.channels)

        self.leacky = nn.LeakyReLU(0.2,inplace=True)
        self.avg_pool = nn.AvgPool2d(kernel_size=2,stride=2)

        self.prog_blocks, self.rgb_layers = nn.ModuleList(), nn.ModuleList()

        for i in range(len(self.channels)):
            self.rgb_layers.append(WSConv2d(self.img_channels,self.channels[i],kernel_size=1,stride=1,padding=0))

        for i in range(len(self.channels)-1):
            self.prog_blocks.append(ConvBlock(self.channels[i+1],self.channels[i],use_pixel_norm=False))

        # self.prog_blocks, self.rgb_layers = nn.ModuleList(), nn.ModuleList()
        # for i in range(len(self.channels),1,-1):
        #     self.prog_blocks.append(ConvBlock(self.channels[i],self.channels[i-1],use_pixel_norm=False))
        #     self.rgb_layers.append(WSConv2d(self.img_channels,self.channels[i-1],kernel_size=1,stride=1,padding=0))

        
        # self.initial_rgb = WSConv2d(self.img_channels,self.channels[-1],kernel_size=1,stride=1,padding=0)
        # self.rgb_layers.append(self.initial_rgb)
        # self.avg_pool = nn.AvgPool2d(kernel_size=2,stride=2)

        self.final_block = nn.Sequential(
            WSConv2d(self.channels[0]+1,self.channels[0],kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(0.2,inplace=True),
            WSConv2d(self.channels[0],self.channels[0],kernel_size=4,stride=1,padding=0),
            nn.LeakyReLU(0.2,inplace=True),
            WSConv2d(self.channels[0],1,kernel_size=1,stride=1,padding=0),
        )

    def fade_in(self,alpha,downscaled,out):
        return alpha *out + (1-alpha)*downscaled

    def minibatch_std(self, x):
        batch_statistics =torch.std(x,dim=0).mean().repeat(x.shape[0],1,x.shape[2],x.shape[3])
        return torch.cat([x,batch_statistics],dim=1)
    
    def forward(self, x,alpha,steps):
            
        out = self.rgb_layers[steps](x)

        if steps !=0:
            out = self.prog_blocks[steps-1](out)
            out = self.avg_pool(out)
            if alpha <1.0:
                downsampled = self.avg_pool(x)
                downsampled = self.rgb_layers[steps-1](downsampled)

                out = self.fade_in(alpha,downsampled,out)

            for s in range(steps-2,-1,-1):
                out = self.prog_blocks[s](out)
                out = self.avg_pool(out)

        out = self.minibatch_std(out)
        out = self.final_block(out)
        return out.view(out.shape[0],-1)
    


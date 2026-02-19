import torch.nn as nn

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        # Kaiming Normal is standard for ReLU/LeakyReLU
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
        # Initialize scale (gamma) to 1 and offset (beta) to 0
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

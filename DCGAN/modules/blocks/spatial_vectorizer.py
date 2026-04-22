import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .eq_lr_layers import EQLRConv2d

class SpatialVectorizer(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 number_of_components,
                   degrees_of_freedom):
        super().__init__()
        self.number_of_components = number_of_components
        self.degrees_of_freedom = degrees_of_freedom
        self.c_prime = out_channels
        
        # 1. Use your EQLR convolution
        self.router = EQLRConv2d(
            in_channels=in_channels, 
            out_channels=self.number_of_components * self.degrees_of_freedom, 
            kernel_size=1, 
            stride=1, 
            padding=0
        )
  


        
        # 2. Override the N(0, 1) initialization to 0.0
        # This ensures the Softmax starts uniformly (1/m) to prevent early collapse,
        # while EQLR still equalizes the gradient steps!
        with torch.no_grad():
            self.router.weight.fill_(0.0)
            if self.router.bias is not None:
                self.router.bias.fill_(0.0)
        
        # 3. The Trainable Feature Vectors
        # (Since you use EQLR, you might also want to initialize these as N(0,1) 
        # and scale them dynamically, or just use the scaled initialization below)
        self.shared_vectors = nn.Parameter(
            torch.randn(self.number_of_components, self.degrees_of_freedom, self.c_prime) / math.sqrt(self.c_prime)
        )
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input feature map of shape (b, c, h, w)
        Returns:
            torch.Tensor: Output tensor of shape (b, n, c', h, w)
        """
        b, c, h, w = x.shape
        
        # Step 1: Generate routing map
        # Output shape: (b, n * m, h, w)
        routing_map = self.router(x)
        
        # Step 2: Reshape to separate components and vectors
        # Output shape: (b, n, m, h, w)
        routing_map = routing_map.view(b, self.number_of_components, self.degrees_of_freedom, h, w)
        
        # Step 3: Apply Softmax across the 'm' dimension (index 2)
        # This makes the m weights sum to 1 for every (component, pixel) pair
        routing_weights = F.softmax(routing_map, dim=2)
        
        # Step 4: Compute the linear combination using einsum
        # routing_weights shape: (b, n, m, h, w)
        # shared_vectors shape:  (n, m, c') -> 'c' in einsum
        # Output shape:          (b, n, c', h, w)
        out = torch.einsum('bnmhw,nmc->bnchw', routing_weights, self.shared_vectors)
        
        return out
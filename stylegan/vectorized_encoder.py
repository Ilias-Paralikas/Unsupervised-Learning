
import torch 
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .conventional_blocks import Vectorizer



class VectorisedEncoder(nn.Module):
    def __init__(self, 
                z_dim,
                channels,
                number_of_vectorizers,
                vector_dim,
                degrees_of_freedom,
                vectorizer_linear_dim,
                input_shape =(256,256),
                img_channels=3,
                block_depth=2,
                residual=True,
                use_norm=True):
        super().__init__()
        self.z_dim = z_dim
        self.channels = channels.copy()
        self.number_of_vectorizers = number_of_vectorizers
        self.vector_dim = vector_dim
        self.vectorizer_linear_dim = vectorizer_linear_dim.copy()
        self.encoder = Encoder(z_dim=z_dim,
                               channels=self.channels,
                               img_channels=img_channels,
                               block_depth=block_depth,
                               residual=residual,
                               use_norm=use_norm)

        dummy_input = torch.randn(1,img_channels,*input_shape)
        with torch.no_grad():
            dummy_ouput = self.encoder(dummy_input)
            vectorizer_input_dim = dummy_ouput.shape[1]

        self.vectorizers = nn.ModuleList([
            Vectorizer(in_neuroes=vectorizer_input_dim,
                       vector_dim=vector_dim,
                       degrees_of_freedom=degrees_of_freedom,
                       linear_layer_dim=self.vectorizer_linear_dim)
            for i in range(number_of_vectorizers)
        ])

    def forward(self, x):
        b_size = x.shape[0]
        x = self.encoder(x)
        vectors= torch.zeros(b_size,self.number_of_vectorizers,self.vector_dim).to(x.device)
        for i in range(len(self.vectorizers)):
            vectors[:,i] = self.vectorizers[i](x)
        return vectors

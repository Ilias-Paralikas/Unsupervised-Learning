import torch
import torch.nn as nn

from model.modules import Decoder
from model.modules import Vectorizer

class ReconstructionDecoder(nn.Module):
    def __init__(self,
                 out_channels=3,
                 vectorizer_in_neurons=1024,
                 number_of_components=4,
                 degrees_of_freedom=32,
                 vector_dim=256,
                 vectorizer_linear_layer_dim =[1024],
                 vectorizer_norm=nn.LayerNorm,
                 vectorizer_activation=nn.ReLU(inplace=True),
                 decoder_channels=[2048,1024,512,256,128,64,32,16],
                 first_conv_size= 4,
                 double_conv=True,
                 decoder_norm=nn.BatchNorm2d,
                 decoder_activation=nn.ReLU(inplace=True),
                 decoder_output_layer_activation=nn.Sigmoid()):
        super().__init__()
        # vectorizer parameters
        self.number_of_components = number_of_components
        self.degrees_of_freedom = degrees_of_freedom
        self.vector_dim = vector_dim
        self.vectorizer_linear_layer_dim= vectorizer_linear_layer_dim.copy()
        self.vectorizer_norm = vectorizer_norm
        self.vectorizer_activation = vectorizer_activation
        # decoder parameters
        self.out_channels = out_channels
        self.vectorizer_in_neurons = vectorizer_in_neurons
        self.decoder_channels = decoder_channels.copy()
        self.first_conv_size = first_conv_size
        self.double_conv = double_conv
        self.decoder_norm = decoder_norm
        self.decoder_activation = decoder_activation
        self.decoder_output_layer_activation = decoder_output_layer_activation

        self.vectorizers = nn.ModuleList([Vectorizer(in_neurons=self.vectorizer_in_neurons,
                                                     vector_dim=self.vector_dim,
                                                     degrees_of_freedom=self.degrees_of_freedom,
                                                     linear_layer_dim=self.vectorizer_linear_layer_dim,
                                                     norm=self.vectorizer_norm,
                                                     activation=self.vectorizer_activation) 
                                                     for _ in range(self.number_of_components)])


        self.decoder = Decoder(bottleneck_dim=self.vector_dim,
                               out_channels=self.out_channels,
                               channels=self.decoder_channels,
                               output_layer_activation=self.decoder_output_layer_activation,
                               double_conv=self.double_conv,
                               first_conv_size=self.first_conv_size,
                               activation=self.decoder_activation,
                               norm=self.decoder_norm)
          
    # def forward(self, x):
    #     batch_size = x.shape[0]
    #     # pass the encoder output through each vectorizer
    #     component_vectors = torch.stack([v(x) for v in self.vectorizers], dim=1)
    #    # Use reshape to avoid Contiguity errors
    #     flat_shape = (batch_size * self.number_of_components, -1, 1, 1)
    #     vectors = component_vectors.reshape(flat_shape)

    #     x = self.decoder(vectors)

    #     # Use -1 to let PyTorch infer the spatial dimensions (H, W) 
    #     # so the code doesn't break if you change the decoder resolution
    #     x = x.reshape(batch_size, self.number_of_components, self.out_channels, x.shape[-2], x.shape[-1])
    #     return x, component_vectors
    def forward(self, x):
        component_vectors = torch.stack([v(x) for v in self.vectorizers], dim=1)

        reconstructions = []
        for i in range(self.number_of_components):
            v= component_vectors[:,i].unsqueeze(-1).unsqueeze(-1)
            reconstructions.append(self.decoder(v)) 
        reconstructions = torch.stack(reconstructions, dim=1)
        return reconstructions, component_vectors
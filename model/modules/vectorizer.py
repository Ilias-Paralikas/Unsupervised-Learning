import torch.nn as nn
import torch

from .blocks import LinearNeuralNetwork

class Vectorizer(nn.Module):
    def __init__(self,
                 in_neurons,
                 vector_dim,
                 degrees_of_freedom,
                linear_layer_dim,
                norm,
                activation):
        super().__init__()
        self.in_neurons = in_neurons    
        self.vector_dim = vector_dim
        self.degrees_of_freedom = degrees_of_freedom
        self.linear_layer_dim = linear_layer_dim.copy()
        self.norm = norm
        self.activation = activation

        
        self.linear_layer_dim.append(self.degrees_of_freedom)



        self.linear = LinearNeuralNetwork(in_neurons=self.in_neurons,
                                          out_neurons=self.degrees_of_freedom,
                                          layer_dims=self.linear_layer_dim,
                                          norm=self.norm,
                                          activation=self.activation)  

        self.vectors = nn.Parameter(torch.rand(self.degrees_of_freedom, self.vector_dim))

    def forward(self, x):
            # 1. Get coefficients from encoder
            a = self.linear(x) 
            
            # 2. Normalize the dictionary vectors along the vector_dim axis
            # This makes the "size" of each concept exactly 1.0
            norm_vectors = self.vectors / (self.vectors.norm(p=2, dim=1, keepdim=True) + 1e-8)
            
            # 3. Reconstruct: (Batch, DOF) @ (DOF, Vector_Dim)
            x = torch.matmul(a, norm_vectors)
            return x

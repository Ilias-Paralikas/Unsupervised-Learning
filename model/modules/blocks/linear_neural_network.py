import torch 
import torch.nn as nn

class LinearNeuralNetwork(nn.Module):
    def __init__(self,
                 in_neurons,
                 out_neurons,
                 layer_dims,
                 norm,
                 activation):
        super().__init__()
        self.in_neurons = in_neurons
        self.out_neurons = out_neurons
        self.layer_dims = layer_dims.copy()
        self.norm = norm
        self.activation = activation

        self.layer_dims.append(self.out_neurons)
        linear_layer =nn.ModuleList()
        linear_layer.append(nn.Linear(self.in_neurons,self.layer_dims[0]))
        for i in range(len(self.layer_dims)-1):
            linear_layer.append(self.norm(self.layer_dims[i]))
            linear_layer.append(self.activation)
            linear_layer.append(nn.Linear(self.layer_dims[i], self.layer_dims[i+1]))

        self.linear = nn.Sequential(*linear_layer)
    def forward(self, x):
        x = self.linear(x)
        return x
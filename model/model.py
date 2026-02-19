import torch
import torch.nn as nn
import torch.nn.functional as F

from .reconstruction_decoder import ReconstructionDecoder
from .segmentation_decoder import SegmentationDecoder
from .modules import Encoder

class Model(nn.Module):
        def __init__(self,
                        in_channels=3, 
                        # encoder parameters
                        encoder_channels= [32, 64, 128, 256, 512, 1024,2048],
                        encoder_norm=nn.BatchNorm2d,
                        encoder_activation=nn.ReLU(inplace=True),
                        input_size = (512,512),
                        # reconstruction decoder parameters
                        number_of_components=4,
                        degrees_of_freedom=32,
                        vector_dim=256,
                        vectorizer_linear_layer_dim =[1024],
                        vectorizer_norm=nn.LayerNorm,
                        vectorizer_activation=nn.ReLU(inplace=True),
                        rec_dec_channels=[2048,1024,512,256,128,64,32,16],
                        rec_dec_first_conv_size= 4,
                        rec_dec_double_conv=True,
                        rec_dec_norm=nn.BatchNorm2d,
                        rec_dec_activation=nn.ReLU(inplace=True),
                        rec_dec_output_layer_activation=nn.Sigmoid(),
                        # segmentation decoder parameters
                        seg_dec_linear_in_neurons=1024,
                        seg_dec_linear_out_neurons=256,
                        seg_dec_linear_layer_dim=[],
                        seg_dec_linear_norm=nn.LayerNorm,
                        seg_dec_linear_activation=nn.ReLU(inplace=True),
                        seg_dec_channels=[2048,1024,512,256,128,64,32,16],
                        seg_dec_first_conv_size= 4,
                        seg_dec_double_conv=True,
                        seg_dec_norm=nn.BatchNorm2d,
                        seg_dec_activation=nn.ReLU(inplace=True),
                        seg_dec_output_layer_activation=nn.Softmax(dim=1)):

                super().__init__()
                self.in_channels = in_channels
                # encoder parameters
                self.encoder_channels = encoder_channels
                self.encoder_norm = encoder_norm
                self.encoder_activation = encoder_activation
                self.input_size = input_size
                # reconstruction decoder parameters
                self.number_of_components = number_of_components
                self.degrees_of_freedom = degrees_of_freedom
                self.vector_dim = vector_dim
                self.vectorizer_linear_layer_dim= vectorizer_linear_layer_dim.copy()
                self.vectorizer_norm = vectorizer_norm
                self.vectorizer_activation = vectorizer_activation
                self.rec_dec_channels = rec_dec_channels.copy()
                self.rec_dec_first_conv_size = rec_dec_first_conv_size
                self.rec_dec_double_conv = rec_dec_double_conv
                self.rec_dec_norm = rec_dec_norm
                self.rec_dec_activation = rec_dec_activation
                self.rec_dec_output_layer_activation = rec_dec_output_layer_activation
        

                # segmentation decoder parameters
                self.seg_dec_linear_in_neurons = seg_dec_linear_in_neurons  
                self.seg_dec_linear_out_neurons = seg_dec_linear_out_neurons
                self.seg_dec_linear_layer_dim = seg_dec_linear_layer_dim.copy()
                self.seg_dec_linear_norm = seg_dec_linear_norm
                self.seg_dec_linear_activation = seg_dec_linear_activation
                self.seg_dec_channels = seg_dec_channels.copy()
                self.seg_dec_first_conv_size = seg_dec_first_conv_size
                self.seg_dec_double_conv = seg_dec_double_conv
                self.seg_dec_norm = seg_dec_norm
                self.seg_dec_activation = seg_dec_activation
                self.seg_dec_output_layer_activation = seg_dec_output_layer_activation
                
                self.encoder = Encoder(in_channels=self.in_channels,
                                channels=self.encoder_channels,
                                norm=self.encoder_norm,
                                activation=self.encoder_activation,
                                input_size=self.input_size)
                with torch.no_grad():
                        dummy_x = torch.randn(1,self.in_channels,*self.input_size)
                        dummy_out = self.encoder(dummy_x)
                        encoder_output_shape = dummy_out.shape[1]

                self.reconstruction_decoder = ReconstructionDecoder(out_channels=self.in_channels,
                                                                vectorizer_in_neurons=encoder_output_shape,
                                                                number_of_components=self.number_of_components,
                                                                degrees_of_freedom=self.degrees_of_freedom,
                                                                vector_dim=self.vector_dim,
                                                                vectorizer_linear_layer_dim=self.vectorizer_linear_layer_dim,
                                                                vectorizer_norm=self.vectorizer_norm,
                                                                vectorizer_activation=self.vectorizer_activation,
                                                                decoder_channels=self.rec_dec_channels,
                                                                first_conv_size=self.rec_dec_first_conv_size,
                                                                double_conv=self.rec_dec_double_conv,
                                                                decoder_norm=self.rec_dec_norm,
                                                                decoder_activation=self.rec_dec_activation,
                                                                decoder_output_layer_activation=self.rec_dec_output_layer_activation)

                
                self.segmentation_decoder = SegmentationDecoder(out_channels=self.number_of_components,
                                                                linear_in_neurons=encoder_output_shape,
                                                                linear_out_neurons=self.seg_dec_linear_out_neurons,
                                                                linear_layer_dim=self.seg_dec_linear_layer_dim,
                                                                linear_norm=self.seg_dec_linear_norm,
                                                                linear_activation=self.seg_dec_linear_activation,
                                                                decoder_channels=self.seg_dec_channels,
                                                                first_conv_size=self.seg_dec_first_conv_size,
                                                                double_conv=self.seg_dec_double_conv,
                                                                decoder_norm=self.seg_dec_norm,
                                                                decoder_activation=self.seg_dec_activation,
                                                                decoder_output_layer_activation=self.seg_dec_output_layer_activation)
        
        def forward(self, x):
                encoder_output = self.encoder(x)
                reconstructions,vectors = self.reconstruction_decoder(encoder_output)
                segmentations = self.segmentation_decoder(encoder_output)
                return reconstructions, segmentations,vectors

        @staticmethod
        def reconstruct_image(reconstructions, segmentations, binary_masks=True):
                if binary_masks:
                        # Get the winning indices
                        indices = torch.argmax(segmentations, dim=1)

                        # Convert to one-hot and ensure it's the same float type as reconstructions
                        # We permute to get (B, N, H, W)
                        segmentations = F.one_hot(indices, num_classes=segmentations.shape[1])
                        segmentations = segmentations.permute(0, 3, 1, 2).to(reconstructions.dtype)

                # The unsqueeze(2) adds the channel dimension for broadcasting
                masked_reconstructions = reconstructions * segmentations.unsqueeze(2)

                # Summing over the N dimension to collapse
                return masked_reconstructions.sum(dim=1)
import torch
import torch.nn as nn
import torch.nn.functional as F


from .modules.blocks.eq_lr_layers import EQLRConv2d
from .modules.blocks import ConvBlock

from .encoder import Encoder
from .decoder import Decoder


from .modules.blocks import SpatialVectorizer

class VectorizedUNet(nn.Module):
    def __init__(self,
                 encoder_channels,
                 decoder_channels,
                 vectorizer_output_channels=None,
                 z_dim=256,
                 number_of_components=4,   
                 degrees_of_freedom=None,     
                 grid_sizes=None,
                 block_depth=2,
                 residual=True,
                 in_channels=3,
                 out_channels=3,
                 use_norm=False,
                 cut_connections=0):
        super().__init__()

        if vectorizer_output_channels is not None and cut_connections!=0:
            raise ValueError('''You passed both a vectorizer output channels list
                             as well as a cut connections index, which will be overwritten.
                             Chances are that this is not intentional
                             Make sure you only pass one of the two''')
        self.number_of_components= number_of_components

        if grid_sizes is None:
            self.grid_sizes = [4]*(len(encoder_channels))
        else:
            self.grid_sizes = grid_sizes.copy()

        if degrees_of_freedom is None:
            self.degrees_of_freedom = [8]*(len(encoder_channels))
        else:
            self.degrees_of_freedom = degrees_of_freedom.copy()

        self.cut_connections = cut_connections
        self.encoder_channels  = encoder_channels.copy()
        # ── Encoder ────────────────────────────────────────────────────
        self.encoder = Encoder(channels=self.encoder_channels,
                               z_dim=z_dim,
                               block_depth=block_depth,
                               residual=residual,
                               img_channels=in_channels,
                               use_norm=use_norm)   
        

        # ── Vectorizers ────────────────────────────────────────────────────
        self.bottleneck_vectorizer  =SpatialVectorizer(
                in_channels=z_dim, 
                out_channels=z_dim, 
                number_of_components=self.number_of_components, 
                degrees_of_freedom=self.degrees_of_freedom[0],
                grid_size=self.grid_sizes[0]
        )
    
        self.vectorizer_input_channels = encoder_channels.copy()
        if vectorizer_output_channels is None:
            self.vectorizer_output_channels = self.encoder_channels[:-(cut_connections+1)]
        else: 
            self.vectorizer_output_channels = vectorizer_output_channels.copy()

        self.vectorizers = nn.ModuleList()
        for i in range(len(self.vectorizer_output_channels)):
            # Vectorizer outputs shape (b, n, c', h, w) 
            self.vectorizers.append(SpatialVectorizer(
                in_channels=self.vectorizer_input_channels[i], 
                out_channels=self.vectorizer_output_channels[i], 
                number_of_components=self.number_of_components, 
                degrees_of_freedom=self.degrees_of_freedom[i+1],
                grid_size=self.grid_sizes[i+1]
            ))

     
        # ── Decoder ────────────────────────────────────────────────────
        self.decoder_channels = decoder_channels.copy()
        self.skip_channels = self.vectorizer_output_channels.copy()
        self.decoder = Decoder(channels=self.decoder_channels,
                               skip_channels=self.skip_channels,
                               z_dim=z_dim,
                               block_depth=block_depth,
                               residual=residual,
                               img_channels=out_channels,
                               use_norm=use_norm)
        
        # ── To RGB ────────────────────────────────────────────────────
        self.to_rgb_layers = nn.ModuleList()
        # We need a to_rgb layer for every feature map returned by the Decoder
        # The Decoder returns features from `from_noise` (channels[0]) 
        # and then from every block (channels[1], channels[2], etc.) except the very last one.
        
        for c in self.decoder_channels[:-1]: 
            self.to_rgb_layers.append(
                nn.Sequential(
                ConvBlock(in_channels=c,
                      out_channels=c,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      use_norm=use_norm),
                ConvBlock(in_channels=c,
                      out_channels=c,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      use_norm=use_norm),
                EQLRConv2d(in_channels=c, 
                           out_channels=out_channels, # e.g., 3 for RGB or 1 for Grayscale
                           kernel_size=1, 
                           stride=1, 
                           padding=0)
            )
            )
   


    def forward(self, 
                x, 
                seg_masks, 
                return_intermediates=True):
        def combine(reconstructions,segmentations):
            pooled_masks = segmentations.unsqueeze(2)
            blended_skip = (reconstructions * pooled_masks).sum(dim=1)
            return blended_skip
        
        def mask_vectorizer_output(vec_out, seg_masks):
            b, n, c_prime, h, w = vec_out.shape
            pooled_masks = F.adaptive_avg_pool2d(seg_masks, (h, w)) 
            
            blended_skip = combine(vec_out, pooled_masks)
            return blended_skip
     

        # encoder
        bottleneck, skip_connections = self.encoder(x, return_features=True)

        vectorized_bottleneck = self.bottleneck_vectorizer (bottleneck)
        vectorized_bottleneck = mask_vectorizer_output(vectorized_bottleneck, seg_masks)

        active_skips = skip_connections[self.cut_connections:]
        active_skips.reverse()
        blended_skips = []
        for skip_feature, vectorizer in zip(active_skips, self.vectorizers):
            vec_out = vectorizer(skip_feature)
            blended_skip = mask_vectorizer_output(vec_out, seg_masks)
            blended_skips.append(blended_skip)


        blended_skips.reverse()
        final_img, intermediate_features = self.decoder(
            vectorized_bottleneck, 
            skip_connections=blended_skips, 
            get_intermediate_features=True
        )
        
        if not return_intermediates:
            return final_img
            
        # Project all intermediate features to RGB images
        all_imgs = []
        for feature, to_rgb_layer in zip(intermediate_features, self.to_rgb_layers):
            rgb = to_rgb_layer(feature)
            rgb = torch.tanh(rgb)
            all_imgs.append(rgb)
            
        # 2. Append the final 256x256 image to the end of the list!
        final_img= torch.tanh(final_img)
        all_imgs.append(final_img)
            
        return all_imgs
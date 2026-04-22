import torch
import torch.nn as nn
import torch.nn.functional as F


from .modules.blocks.eq_lr_layers import EQLRConv2d

from .encoder import Encoder
from .decoder import Decoder


from .modules.blocks import SpatialVectorizer

class VectorizedUNet(nn.Module):
    def __init__(self,
                 channels,
                 vectorizer_output_channels=None,
                 number_of_components=4,   
                 degrees_of_freedom=8,     
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
        self.degrees_of_freedom = degrees_of_freedom

        self.cut_connections = cut_connections
        self.encoder_channels  = channels.copy()
        # ── Encoder ────────────────────────────────────────────────────
        self.encoder = Encoder(channels=self.encoder_channels,
                               z_dim=self.encoder_channels[0],
                               block_depth=block_depth,
                               residual=residual,
                               img_channels=in_channels,
                               use_norm=use_norm)   
        

        # ── Vectorizers ────────────────────────────────────────────────────
        self.vectorizer_input_channels = channels.copy()
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
                degrees_of_freedom=self.degrees_of_freedom
            ))
        
        # ── Decoder ────────────────────────────────────────────────────
        self.decoder_channels = channels.copy()
        self.skip_channels = self.vectorizer_output_channels.copy()
        self.decoder = Decoder(channels=self.decoder_channels,
                               skip_channels=self.skip_channels,
                               z_dim=self.decoder_channels[0],
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
                EQLRConv2d(in_channels=c, 
                           out_channels=out_channels, # e.g., 3 for RGB or 1 for Grayscale
                           kernel_size=1, 
                           stride=1, 
                           padding=0)
            )
   


    def forward(self, 
                x, 
                seg_masks, 
                return_intermediates=True):
        def combine(reconstructions,segmentations):
            pooled_masks = segmentations.unsqueeze(2)
            blended_skip = (reconstructions * pooled_masks).sum(dim=1)
            return blended_skip
     
        x, skip_connections = self.encoder(x, return_features=True)
        active_skips = skip_connections[self.cut_connections:]
        active_skips.reverse()
        blended_skips = []
        for skip_feature, vectorizer in zip(active_skips, self.vectorizers):
            vec_out = vectorizer(skip_feature)
            b, n, c_prime, h, w = vec_out.shape
            
            pooled_masks = F.adaptive_avg_pool2d(seg_masks, (h, w)) 
            
            blended_skip = combine(vec_out, pooled_masks)
            blended_skips.append(blended_skip)


        blended_skips.reverse()
        final_img, intermediate_features = self.decoder(
            x, 
            skip_connections=blended_skips, 
            get_intermediate_features=True
        )
        
        if not return_intermediates:
            return final_img
            
        # Project all intermediate features to RGB images
        intermediate_imgs = []
        for feature, to_rgb in zip(intermediate_features, self.to_rgb_layers):
            img = to_rgb(feature)
            img = torch.tanh(img) 
            intermediate_imgs.append(img)
            
        # intermediate_imgs contains e.g., [img_4x4, img_8x8, img_16x16, ...]
        # final_img is the full resolution image (e.g., img_256x256)
        return final_img, intermediate_imgs
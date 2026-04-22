import torch
import torch.nn as nn
import torch.nn.functional as F

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
        

        # --- Vectorizers ---
        if vectorizer_output_channels is None:
            self.vectorizer_output_channels = self.encoder_channels[:-(cut_connections+1)]
        else: 
            self.vectorizer_output_channels = vectorizer_output_channels.copy()

        self.vectorizers = nn.ModuleList()
        
        for i in range(len(self.vectorizer_output_channels)):
            # Vectorizer outputs shape (b, n, c', h, w) 
            self.vectorizers.append(SpatialVectorizer(
                in_channels=self.encoder_channels[i], 
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
        
   


    def forward(self, x, seg_masks):
        def combine(reconstructions,segmentations):
            pooled_masks = segmentations.unsqueeze(2)
            blended_skip = (reconstructions * pooled_masks).sum(dim=1)
            return blended_skip
     
        x, skip_connections = self.encoder(x, return_features=True)
        active_skips = skip_connections[self.cut_connections:]
        
        blended_skips = []
        for skip_feature, vectorizer in zip(active_skips, self.vectorizers):
            
            vec_out = vectorizer(skip_feature)
            b, n, c_prime, h, w = vec_out.shape
            
            pooled_masks = F.adaptive_avg_pool2d(seg_masks, (h, w)) 
            
            blended_skip = combine(vec_out, pooled_masks)
            blended_skips.append(blended_skip)
            
        out = self.decoder(x, skip_connections=blended_skips)
        
        return out
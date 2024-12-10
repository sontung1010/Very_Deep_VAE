from torch import nn
import torch
from vae_block.vae_helpers import prepare_string, from_parameter_get_width
from vae_block.vae_helpers import convolution_3x3
import numpy as np
from vae_block.base_block import Block
from vae_block.DmolNet import DmolNet, HModule

## Defining the Encoder
# This is the implementation of the encoder structure.
# Encoder structure was fairly straight forward.
# =======================================================================
class Encoder(HModule):

    def build(self):
        H = self.H  # Hyperparameters settings
        # Initial convolution layer to process input image channels into the initial width
        self.initial_conv = convolution_3x3(H.image_channels, H.width)

        self.widths = from_parameter_get_width(H.width, H.custom_width_str)
        #print(type(self.widths))
        # List to store the encoding blocks
        encoder_block_list = []
        # Parse the block definitions string to get resolution and down-sampling settings
        blockstr = prepare_string(H.enc_blocks)
        
        for res, down_rate in blockstr:
            # Use 3x3 convolutions only for resolutions larger than 2x2
            if res > 2:
                use_3x3 = True
            else: 
                use_3x3 = False
            # Append a Block to the encoder with specified parameters
            encoder_block_list.append(
                Block(
                    self.widths[res],  # Input width at this resolution
                    int(self.widths[res] * H.bottleneck_multiple),  # Bottleneck width
                    self.widths[res],  # Output width
                    down_sample_rate=down_rate,  # Optional down-sampling rate
                    use_residual=True,  # Enable residual connections
                    use_3x3=use_3x3  # Whether to use 3x3 convolutions
                )
            )
            # print(len(encoder_block_list))
        
        # Normalize the initialization of the final layer in each block to prevent high variance
        n_blocks = len(blockstr)
        for b in encoder_block_list:
            b.conv4.weight.data *= np.sqrt(1 / n_blocks)  # Scale weights inversely to the number of blocks
        # Store the encoding blocks as a ModuleList to register them as submodules
        self.enc_blocks = nn.ModuleList(encoder_block_list)

    
    def padding_channel(self, t, width):
        dim1, dim2, dim3, dim4 = t.shape  # Extract shape of the input tensor
        empty = torch.zeros(dim1, width, dim3, dim4, device=t.device)  # Create a tensor with target width
        empty[:, :dim2, :, :] = t 
        return empty

    def forward(self, x):
        # (batch, height, width, channels) to (batch, channels, height, width)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.initial_conv(x)
        
        # Dictionary to store intermediate activations at each resolution
        activations = {}
        activations[x.shape[2]] = x 
        
        for block in self.enc_blocks:
            x = block(x) 
            res = x.shape[2]
            if x.shape[1] == self.widths[res]:
                updated_x = x
            else:
                updated_x = self.padding_channel(x, self.widths[res])
            x = updated_x
            activations[res] = x
        # return the dictionary of activations at different resolutions
        return activations

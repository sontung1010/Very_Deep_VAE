from torch import nn
import torch
from vae_block.vae_helpers import parse_layer_string, get_width_settings
from vae_block.vae_helpers import get_3x3
import numpy as np
from vae_block.base_block import Block
from vae_block.DmolNet import DmolNet, HModule

## Defining the Encoder
# =======================================================================

class Encoder(HModule):

    def build(self):
        H = self.H  # Hyperparameters or configuration settings
        # Initial convolution layer to process input image channels into the initial width
        self.in_conv = get_3x3(H.image_channels, H.width)
        # Determine the width settings for different resolutions based on provided configuration
        self.widths = get_width_settings(H.width, H.custom_width_str)
        # List to store the encoding blocks
        enc_blocks = []
        # Parse the block definitions string to get resolution and down-sampling settings
        blockstr = parse_layer_string(H.enc_blocks)
        
        for res, down_rate in blockstr:
            # Use 3x3 convolutions only for resolutions larger than 2x2
            use_3x3 = res > 2
            # Append a Block to the encoder with specified parameters
            enc_blocks.append(
                Block(
                    self.widths[res],  # Input width at this resolution
                    int(self.widths[res] * H.bottleneck_multiple),  # Bottleneck width
                    self.widths[res],  # Output width
                    down_sample_rate=down_rate,  # Optional down-sampling rate
                    use_residual=True,  # Enable residual connections
                    use_3x3=use_3x3  # Whether to use 3x3 convolutions
                )
            )
        
        # Normalize the initialization of the final layer in each block to prevent high variance
        n_blocks = len(blockstr)
        for b in enc_blocks:
            b.conv4.weight.data *= np.sqrt(1 / n_blocks)  # Scale weights inversely to the number of blocks
        # Store the encoding blocks as a ModuleList to register them as submodules
        self.enc_blocks = nn.ModuleList(enc_blocks)

    
    def padding_channel(self, t, width):
        dim1, dim2, dim3, dim4 = t.shape  # Extract shape of the input tensor
        empty = torch.zeros(dim1, width, dim3, dim4, device=t.device)  # Create a tensor with target width
        empty[:, :dim2, :, :] = t  # Copy the original tensor's channels into the new tensor
        return empty


    def forward(self, x):
        # Permute input tensor from (batch, height, width, channels) to (batch, channels, height, width)
        x = x.permute(0, 3, 1, 2).contiguous()
        
        # Pass through the initial convolution layer
        x = self.in_conv(x)
        
        # Dictionary to store intermediate activations at each resolution
        activations = {}
        activations[x.shape[2]] = x  # Store the activation for the initial resolution
        
        # Process the input through the sequence of encoder blocks
        for block in self.enc_blocks:
            x = block(x)  # Apply the block
            # Get the current resolution (height/width)
            res = x.shape[2]
            # Ensure the output has the expected width by padding channels if necessary
            x = x if x.shape[1] == self.widths[res] else self.padding_channel(x, self.widths[res])
            # Store the activation for the current resolution
            activations[res] = x
        # Return the dictionary of activations at different resolutions
        return activations

from torch import nn
from torch.nn import functional as F
from vae_block.vae_helpers import get_1x1, get_3x3


class Block(nn.Module):
    def __init__(self, input_channels, intermediate_channels, output_channels, down_sample_rate=None, use_residual=False, zero_last=False, use_3x3=True):
        super().__init__()
        self.down_sample_rate = down_sample_rate
        self.use_residual = use_residual
        # 1x1 convolution from input_channels to intermediate_channels
        self.conv1 = get_1x1(input_channels, intermediate_channels)
        # 3x3 convolution (or 1x1 if use_3x3 is False)
        if use_3x3: self.conv2 = get_3x3(intermediate_channels, intermediate_channels)
        else : self.conv2 = get_1x1(intermediate_channels, intermediate_channels)
        # Another 3x3 convolution (or 1x1 if use_3x3 is False)
        if use_3x3 : self.conv3 = get_3x3(intermediate_channels, intermediate_channels)
        else: self.conv3 = get_1x1(intermediate_channels, intermediate_channels)
        # 1x1 convolution to project to output_channels (optionally initializing weights to zero)
        self.conv4 = get_1x1(intermediate_channels, output_channels, zero_weights=zero_last)

    def forward(self, inputs):
        # Apply the series of convolutions with GELU activation in between
        intermediate_output = self.conv1(F.gelu(inputs))
        intermediate_output = self.conv2(F.gelu(intermediate_output))
        intermediate_output = self.conv3(F.gelu(intermediate_output))
        intermediate_output = self.conv4(F.gelu(intermediate_output))
        
        # Add residual connection if specified
        if self.use_residual:final_output = inputs + intermediate_output
        else : final_output = intermediate_output

        # Optionally apply down-sampling using average pooling
        if self.down_sample_rate is not None:
            final_output = F.avg_pool2d(final_output, kernel_size=self.down_sample_rate, stride=self.down_sample_rate)      
        return final_output
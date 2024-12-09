from torch import nn
from torch.nn import functional as F
from vae_block.vae_helpers import convolution_1x1, convolution_3x3


class Block(nn.Module):
    def __init__(self, input_channels, intermediate_channels, output_channels, down_sample_rate=None, use_residual=False, zero_last=False, use_3x3=True):
        super().__init__()

        # 1x1 convolution from input_channels to intermediate_channels
        self.conv1 = convolution_1x1(input_channels, intermediate_channels)
        # 3x3 convolution (or 1x1 if use_3x3 is False)
        if use_3x3 == True :
            self.conv2 = convolution_3x3(intermediate_channels, intermediate_channels)
        else :
            self.conv2 = convolution_1x1(intermediate_channels, intermediate_channels)
        if use_3x3 == True:
            self.conv3 = convolution_3x3(intermediate_channels, intermediate_channels)
        else:
            self.conv3 = convolution_1x1(intermediate_channels, intermediate_channels)
        # 1x1 convolution to project to output_channels (optionally initializing weights to zero)
        self.conv4 = convolution_1x1(intermediate_channels, output_channels, zero_weights=zero_last)

        self.down_sample_rate = down_sample_rate
        self.use_residual = use_residual

    def forward(self, inputs):
        # Apply the series of convolutions with GELU activation in between
        output_1 = self.conv1(F.gelu(inputs))
        output_1 = self.conv2(F.gelu(output_1))
        output_1 = self.conv3(F.gelu(output_1))
        output_1 = self.conv4(F.gelu(output_1))
        
        # Add residual connection if specified
        if self.use_residual == True:
            final_output = inputs + output_1
        else :
            final_output = output_1

        # Optionally apply down-sampling using average pooling
        if self.down_sample_rate is not None:
            final_output = F.avg_pool2d(final_output, kernel_size=self.down_sample_rate, stride=self.down_sample_rate)      
        return final_output
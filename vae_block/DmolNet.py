import torch.nn as nn
import numpy as np
from vae_block.vae_helpers import get_conv, discretized_mix_logistic_loss, sample_from_discretized_mix_logistic

class HModule(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.H = H
        self.build()


class DmolNet(nn.Module):
    def __init__(self, Hyperparameters):
        super().__init__()
        self.H = Hyperparameters
        self.width = Hyperparameters.width
        self.out_conv = get_conv(
            Hyperparameters.width, 
            Hyperparameters.num_mixtures * 10, 
            kernel_size=1, 
            stride=1, 
            padding=0
        )

    def negative_likelihood(self, p_x_when_z, x):
        low_bit_setting = self.H.dataset in ['ffhq_256']
        forward_result = self.forward(p_x_when_z)
        nll_result = discretized_mix_logistic_loss(x=x, l=forward_result, low_bit=low_bit_setting)
        return nll_result

    def forward(self, p_x_when_z):
        conv_output = self.out_conv(p_x_when_z)
        xhat = conv_output.permute(0, 2, 3, 1)
        return xhat

    def get_sample(self, p_x_when_z):
        forward_output = self.forward(p_x_when_z)
        sampled_image = sample_from_discretized_mix_logistic(forward_output, self.H.num_mixtures)
        x_hat = (sampled_image + 1.0) * 127.5
        x_hat_clamped = np.minimum(np.maximum(0.0, x_hat.detach().cpu().numpy()), 255.0)
        final_output = x_hat_clamped.astype(np.uint8)

        return final_output
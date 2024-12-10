from torch import nn
import torch
from torch.nn import functional as F
from vae_block.vae_helpers import prepare_string, from_parameter_get_width
from vae_block.vae_helpers import sample_diag_gaussian, compute_gaussian_kl, convolution_1x1
import numpy as np
from  vae_block.base_block import Block
import itertools
from vae_block.DmolNet import DmolNet, HModule


class DecBlock(nn.Module):
    ## Defining the Decoder
    # =======================================================================
    def __init__(self, H, resolution, mixin, n_blocks):
        super().__init__()
        self.H = H


        self.base = resolution  
        # print(self.base)
        self.mixin = mixin
        # Recieving width settings from hyperparameters
        self.widths = from_parameter_get_width(H.width, H.custom_width_str)
        w_here = self.widths[resolution]  # width for the current resolution
        if resolution > 2:
            use_3x3 = True
        else:
            use_3x3 = False
        conditional_width = int(w_here * H.bottleneck_multiple) 
        self.zdim = H.zdim 
        
        # Encoder block to calculate latent parameters
        output_dim = H.zdim * 2
        self.enc = Block(w_here * 2, conditional_width, output_dim, use_residual=False, use_3x3=use_3x3)
        self.prior = Block(w_here, conditional_width, output_dim + w_here, use_residual=False, use_3x3=use_3x3, zero_last=True)
        
        # Residual block for refining features
        self.resnet = Block(w_here, conditional_width, w_here, use_residual=True, use_3x3=use_3x3)
        self.resnet.conv4.weight.data *= np.sqrt(1 / n_blocks)

        # Projection layer to map latent samples z to the desired width
        self.z_proj = convolution_1x1(H.zdim, w_here)
        self.z_proj.weight.data *= np.sqrt(1 / n_blocks)
        self.z_fn = lambda x: self.z_proj(x)

    def get_sample(self, x, acts):
        # Compute mean and variance for the approximate posterior q
        output = self.enc(torch.cat([x, acts], dim=1))
        split_size = output.size(1) // 2
        q_m, q_v = torch.split(output, split_size, dim=1)
        
        # prior distribution parameters p and additional features xpp
        feats = self.prior(x)
        p_m, p_v, xpp = feats[:, :self.zdim, ...], feats[:, self.zdim:self.zdim * 2, ...], feats[:, self.zdim * 2:, ...]
        
        # updte x with the additional features
        x = x + xpp
        # sample z from the approximate posterior
        z = sample_diag_gaussian(q_m, q_v)
        # compute KL divergence among posterior and the prior
        kl_gaussian = compute_gaussian_kl(q_m, p_m, q_v, p_v)
        
        return z, x, kl_gaussian

    # Unconditional sampling from the prior
    def unconditional_sampling(self, x, t=None, lvs=None):
        ## n, c, h, w = x.shape
        # print(n, c, h, w)
        # print(h,w)

        # Compute prior distribution parameters p and additional features xpp
        # Update x with the additional features
        feats = self.prior(x)
        p_m, p_v, xpp = feats[:, :self.zdim, ...], feats[:, self.zdim:self.zdim * 2, ...], feats[:, self.zdim * 2:, ...]
        x = x + xpp
        # print(x)
        # Use provided latent variables lvs
        if lvs is not None:
            z = lvs
        else:
            # temperature
            if t is not None:
                p_v = p_v + torch.ones_like(p_v) * np.log(t)
            # Sample z from the prior
            z = sample_diag_gaussian(p_m, p_v)
        return z, x


    # Forward pass for the decoder block
    def forward(self, xs, activations, get_latents=False):
        # Get inputs for this block      
        activation = activations[self.base]
        try:
            x = xs[self.base]  # previously computed value for this resolution
        except KeyError:
            # initialize x as zeros with the same shape as acts
            x = torch.zeros_like(activation)

        # Repeat x to match the batch size of acts if necessary
        if activation.shape[0] != x.shape[0]:
            x = x.repeat(activation.shape[0], 1, 1, 1)

        # Add mixed-in resolution if specified
        if self.mixin is not None:
            x = x + F.interpolate(xs[self.mixin][:, :x.shape[1], ...], scale_factor=self.base // self.mixin)
        
        # latent variable z and compute KL divergence
        z, x, kl = self.get_sample(x, activation)
        # Update x
        x = x + self.z_fn(z)
        # Refine features using the residual block
        x = self.resnet(x)
        xs[self.base] = x
        
        # Optionally return latents and KL divergence for downstream processing
        if get_latents:
            return xs, dict(z=z.detach(), kl=kl)
        return xs, dict(kl=kl)

    # Forward pass for unconditional sampling
    def forward_unconditional(self, xs, t=None, lvs=None):
        try:
            x = xs[self.base]
        except KeyError:
            ref = xs[list(xs.keys())[0]] # initialize x as zeros with the appropriate shape
            x = torch.zeros(dtype=ref.dtype, size=(ref.shape[0], self.widths[self.base], self.base, self.base), device=ref.device)
        
        # Add mixed in resolution if specified
        if self.mixin is not None:
            x = x + F.interpolate(xs[self.mixin][:, :x.shape[1], ...], scale_factor=self.base // self.mixin)
        
        z, x = self.unconditional_sampling(x, t, lvs=lvs) # sample latent variable z from the prior
        x = x + self.z_fn(z) # Update x with the latent variable project.
        x = self.resnet(x) # Refine features using the residual block
        xs[self.base] = x
        return xs


# Define the Decoder class, responsible for reconstructing data from hierarchical latent representations
class Decoder(HModule):
    def build(self):
        # Hyperparameters settings
        H = self.H  
        resos = set()
        dec_blocks = []
        
        # Get width settings for different resolutions
        self.widths = from_parameter_get_width(H.width, H.custom_width_str)
        # Parse the block definitions string
        blocks = prepare_string(H.dec_blocks)
        for _, (res, mixin) in enumerate(blocks):
            dec_blocks.append(DecBlock(H, res, mixin, n_blocks=len(blocks))) # apend a decoder block for the given resolution and mixin
            resos.add(res)
        self.dec_blocks = nn.ModuleList(dec_blocks)
         # resolutions
        self.resolutions = sorted(resos)

        # Create bias parameters for each resolution, up to the specified resolution limit
        bias_xs_list = []
        for res in self.resolutions:
            if res <= H.no_bias_above:
                bias_xs_list.append(nn.Parameter(torch.zeros(1, self.widths[res], res, res)))
        self.bias_xs = nn.ParameterList(bias_xs_list)
        
        # output -> maps the final decoder outputs to the model's distribution
        self.out_net = DmolNet(H)
        self.gain = nn.Parameter(torch.ones(1, H.width, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, H.width, 1, 1))
        
        # Lambda function for applying the gain and bias
        self.final_fn = lambda x: x * self.gain + self.bias

    def forward(self, activations, get_latents=False):        
        # Initialize xs with biases for the available resolutions
        xs = {}
        for a in self.bias_xs:
            xs[a.shape[2]] = a

        # List to collect statistics from each block
        statistics = []

        # Pass activations and latent states through each decoder block
        for block in self.dec_blocks:
            xs, block_stats = block(xs, activations, get_latents=get_latents)

            statistics.append(block_stats)
        
        # apply final scaling and bias to the output
        xs[self.H.image_size] = self.final_fn(xs[self.H.image_size])
        # Output image
        return xs[self.H.image_size], statistics

    def forward_unconditional(self, n, temperature=None, y=None):
      
        # Initialize xs with repeated biases, batch size n
        xs = {}
        for bias in self.bias_xs:
            xs[bias.shape[2]] = bias.repeat(n, 1, 1, 1)
        
        # pas through each decoder block for unconditional sampling
        for idx, block in enumerate(self.dec_blocks):
            # handling temperature (each block) if specified
            try:
                temp = temperature[idx]
            except TypeError:
                temp = temperature
            xs = block.forward_unconditional(xs, temp)
        
        # Apply final scaling and bias to the output at the target resolution
        xs[self.H.image_size] = self.final_fn(xs[self.H.image_size])
        return xs[self.H.image_size]

    def forward_with_manual_latent_varialbes(self, n, latent_variables, temperature=None):
        # Forward pass with manually provided latent variables
        xs = {}
        for bias in self.bias_xs:
            xs[bias.shape[2]] = bias.repeat(n, 1, 1, 1)
        
        # pass through each decoder block using provided latent variables
        for block, lvs in itertools.zip_longest(self.dec_blocks, latent_variables):
            xs = block.forward_unconditional(xs, temperature, lvs=lvs)
        
        # Apply final scaling and bias to the output at the target resolution
        xs[self.H.image_size] = self.final_fn(xs[self.H.image_size])
        return xs[self.H.image_size]
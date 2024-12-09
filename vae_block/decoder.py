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
        self.base = resolution  
        self.mixin = mixin
        self.H = H
        
        # Recieving width settings from hyperparameters
        self.widths = from_parameter_get_width(H.width, H.custom_width_str)
        width = self.widths[resolution]  # width for the current resolution
        
        use_3x3 = resolution > 2 
        cond_width = int(width * H.bottleneck_multiple) 
        self.zdim = H.zdim 
        
        # Encoder block to calculate latent parameters
        self.enc = Block(width * 2, cond_width, H.zdim * 2, use_residual=False, use_3x3=use_3x3)
        
        # Prior block to compute prior distribution parameters (mean and variance) and add additional features to `x`
        self.prior = Block(width, cond_width, H.zdim * 2 + width, use_residual=False, use_3x3=use_3x3, zero_last=True)
        
        # Projection layer to map latent samples (z) to the desired width
        self.z_proj = convolution_1x1(H.zdim, width)
        self.z_proj.weight.data *= np.sqrt(1 / n_blocks)
        
        # Residual block for refining features
        self.resnet = Block(width, cond_width, width, use_residual=True, use_3x3=use_3x3)
        self.resnet.conv4.weight.data *= np.sqrt(1 / n_blocks)
        self.z_fn = lambda x: self.z_proj(x)

    def get_sample(self, x, acts):
        # Compute mean and variance for the approximate posterior (q)
        qm, qv = self.enc(torch.cat([x, acts], dim=1)).chunk(2, dim=1)
        
        # Compute prior distribution parameters (p) and additional features (xpp)
        feats = self.prior(x)
        pm, pv, xpp = feats[:, :self.zdim, ...], feats[:, self.zdim:self.zdim * 2, ...], feats[:, self.zdim * 2:, ...]
        
        # Update `x` with the additional features
        x = x + xpp
        # Sample `z` from the approximate posterior
        z = sample_diag_gaussian(qm, qv)
        # Compute KL divergence between the posterior and the prior
        kl = compute_gaussian_kl(qm, pm, qv, pv)
        
        return z, x, kl

    # Unconditional sampling from the prior
    def sample_uncond(self, x, t=None, lvs=None):
        n, c, h, w = x.shape  # Get the shape of the input tensor
        # print(n, c, h, w)

        # Compute prior distribution parameters (p) and additional features (xpp)
        # Update `x` with the additional features
        feats = self.prior(x)
        pm, pv, xpp = feats[:, :self.zdim, ...], feats[:, self.zdim:self.zdim * 2, ...], feats[:, self.zdim * 2:, ...]
        x = x + xpp
        
        # Use provided latent variables (lvs) if available
        if lvs is not None:
            z = lvs
        else:
            # Adjust variance with temperature `t` if specified
            if t is not None:
                pv = pv + torch.ones_like(pv) * np.log(t)
            # Sample `z` from the prior
            z = sample_diag_gaussian(pm, pv)
        
        return z, x

    # Retrieve inputs for the current block from activations and previously computed values
    def get_inputs(self, xs, activations):
        acts = activations[self.base]  # Activation for the current resolution
        try:
            x = xs[self.base]  # Previously computed value for this resolution
        except KeyError:
            # If not available, initialize `x` as zeros with the same shape as `acts`
            x = torch.zeros_like(acts)
        # Repeat `x` to match the batch size of `acts` if necessary
        if acts.shape[0] != x.shape[0]:
            x = x.repeat(acts.shape[0], 1, 1, 1)
        return x, acts

    # Forward pass for the decoder block
    def forward(self, xs, activations, get_latents=False):
        # Get inputs for this block
        x, acts = self.get_inputs(xs, activations)
        
        # Add mixed-in resolution if specified
        if self.mixin is not None:
            x = x + F.interpolate(xs[self.mixin][:, :x.shape[1], ...], scale_factor=self.base // self.mixin)
        
        # Sample latent variable `z` and compute KL divergence
        z, x, kl = self.get_sample(x, acts)
        # Update `x` with the latent variable projection
        x = x + self.z_fn(z)
        # Refine features using the residual block
        x = self.resnet(x)
        xs[self.base] = x
        
        # Optionally return latents and KL divergence for downstream processing
        if get_latents:
            return xs, dict(z=z.detach(), kl=kl)
        return xs, dict(kl=kl)

    # Forward pass for unconditional sampling
    def forward_uncond(self, xs, t=None, lvs=None):
        try:
            x = xs[self.base]  # Retrieve input for the current resolution
        except KeyError:
            ref = xs[list(xs.keys())[0]] # If not available, initialize `x` as zeros with the appropriate shape
            x = torch.zeros(dtype=ref.dtype, size=(ref.shape[0], self.widths[self.base], self.base, self.base), device=ref.device)
        
        # Add mixed-in resolution if specified
        if self.mixin is not None:
            x = x + F.interpolate(xs[self.mixin][:, :x.shape[1], ...], scale_factor=self.base // self.mixin)
        
        z, x = self.sample_uncond(x, t, lvs=lvs) # Sample latent variable `z` from the prior
        x = x + self.z_fn(z) # Update `x` with the latent variable projection
        x = self.resnet(x) # Refine features using the residual block
        xs[self.base] = x # Store the updated `x` in the dictionary
        return xs


# Define the Decoder class, responsible for reconstructing data from hierarchical latent representations
class Decoder(HModule):
    def build(self):
        H = self.H  # Hyperparameters or configuration settings
        resos = set()
        dec_blocks = []
        
        # Get width settings for different resolutions
        self.widths = from_parameter_get_width(H.width, H.custom_width_str)
        
        # Parse the block definitions string to create decoder blocks
        blocks = prepare_string(H.dec_blocks)
        for idx, (res, mixin) in enumerate(blocks):
            dec_blocks.append(DecBlock(H, res, mixin, n_blocks=len(blocks))) # Append a decoder block for the given resolution and mixin
            resos.add(res)  # Add resolution to the set
        
        # Sort the resolutions in ascending order
        self.resolutions = sorted(resos)
        
        # Store decoder blocks as a ModuleList to register them as submodules
        self.dec_blocks = nn.ModuleList(dec_blocks)
        
        # Create bias parameters for each resolution, up to the specified resolution limit
        self.bias_xs = nn.ParameterList([
            nn.Parameter(torch.zeros(1, self.widths[res], res, res)) 
            for res in self.resolutions if res <= H.no_bias_above
        ])
        
        # Output network that maps the final decoder outputs to the model's distribution
        self.out_net = DmolNet(H)
        
        # Gain and bias parameters applied to the final output
        self.gain = nn.Parameter(torch.ones(1, H.width, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, H.width, 1, 1))
        
        # Lambda function for applying the gain and bias
        self.final_fn = lambda x: x * self.gain + self.bias

    def forward(self, activations, get_latents=False):
        # List to collect statistics (e.g., KL divergence) from each decoder block
        stats = []
        
        # Initialize `xs` with biases for the available resolutions
        xs = {a.shape[2]: a for a in self.bias_xs}
        
        # Pass activations and latent states through each decoder block
        for block in self.dec_blocks:
            xs, block_stats = block(xs, activations, get_latents=get_latents)
            stats.append(block_stats)  # Collect statistics from each block
        
        # Apply final scaling and bias to the output at the target resolution
        xs[self.H.image_size] = self.final_fn(xs[self.H.image_size])
        
        # Return the reconstructed image and collected statistics
        return xs[self.H.image_size], stats

    def forward_uncond(self, n, temperature=None, y=None):
        # Unconditional forward pass, typically used for sampling
        
        # Initialize `xs` with repeated biases for the batch size `n`
        xs = {}
        for bias in self.bias_xs:
            xs[bias.shape[2]] = bias.repeat(n, 1, 1, 1)
        
        # Pass through each decoder block for unconditional sampling
        for idx, block in enumerate(self.dec_blocks):
            # Handle per-block temperature if specified
            try:
                temp = temperature[idx]
            except TypeError:
                temp = temperature
            xs = block.forward_uncond(xs, temp)
        
        # Apply final scaling and bias to the output at the target resolution
        xs[self.H.image_size] = self.final_fn(xs[self.H.image_size])
        
        # Return the reconstructed image
        return xs[self.H.image_size]

    def forward_with_manual_latent_varialbes(self, n, latent_variables, temperature=None):
        # Forward pass with manually provided latent variables
        
        # Initialize `xs` with repeated biases for the batch size `n`
        xs = {}
        for bias in self.bias_xs:
            xs[bias.shape[2]] = bias.repeat(n, 1, 1, 1)
        
        # Pass through each decoder block using provided latent variables
        for block, lvs in itertools.zip_longest(self.dec_blocks, latent_variables):
            xs = block.forward_uncond(xs, temperature, lvs=lvs)
        
        # Apply final scaling and bias to the output at the target resolution
        xs[self.H.image_size] = self.final_fn(xs[self.H.image_size])
        return xs[self.H.image_size]
import torch
import numpy as np
from vae_block.decoder import Decoder
from vae_block.encoder import Encoder
from vae_block.DmolNet import HModule


# Define the Variational Autoencoder (VAE) class, which integrates the encoder and decoder
class VAE(HModule):
    def build(self):
        # Initialize the encoder and decoder using the provided configuration (H)
        self.encoder = Encoder(self.H)
        self.decoder = Decoder(self.H)

    def forward(self, x, x_target):
        """
        Input:
        - x: Input data.
        - x_target: Target data (usually the same as `x`, but could differ for augmented data).
        
        Return:
        A dictionary containing:
        - elbo: Evidence Lower Bound, the objective being optimized.
        - distortion: Reconstruction loss (negative log-likelihood of the data given the latents).
        - rate: KL divergence term, regularizing the latent space.
        """
        # Pass the input through the encoder to get activations
        activations = self.encoder.forward(x)
        # Decode the activations to get reconstructed data and KL divergence stats
        px_z, stats = self.decoder.forward(activations)
        # Compute the negative log-likelihood (NLL) of the reconstruction
        distortion_per_pixel = self.decoder.out_net.nll(px_z, x_target)
        # Initialize the rate (KL divergence) term
        rate_per_pixel = torch.zeros_like(distortion_per_pixel)
        # Compute the total number of dimensions in the input
        ndims = np.prod(x.shape[1:])
        
        # Accumulate the KL divergence from all decoder blocks
        for statdict in stats:
            rate_per_pixel += statdict['kl'].sum(dim=(1, 2, 3))  # Sum over spatial dimensions
        
        # Normalize KL divergence by the number of dimensions
        rate_per_pixel /= ndims
        elbo = (distortion_per_pixel + rate_per_pixel).mean() # Compute the loss (elbo)
        
        # Return the computed metrics
        return dict(
            elbo=elbo,  # Evidence Lower Bound
            distortion=distortion_per_pixel.mean(),  # Reconstruction loss
            rate=rate_per_pixel.mean()  # KL divergence regularization term
        )

    def forward_get_latents(self, x):
        """
        Input:
        - x: Input data.
        Return:
        - stats: List of KL divergence statistics and latent samples from decoder blocks.
        """
        # Pass the input through the encoder to get activations
        # Pass activations through the decoder to get latents and their statistics
        activations = self.encoder.forward(x)
        val1, stats = self.decoder.forward(activations, get_latents=True)
        return stats

    def forward_uncond_samples(self, n_batch, t=None):
        """
        Generate unconditional samples from the prior.
        Args:
        - n_batch: Number of samples to generate.
        - t: Temperature parameter for sampling (adjusts variance).
        Returns:
        - Samples from the model's output distribution.
        """
        # Generate samples from the decoder without conditioning on input data
        px_z = self.decoder.forward_uncond(n_batch, t=t)
        # Use the output network to sample from the predicted distribution
        return self.decoder.out_net.sample(px_z)

    def forward_samples_set_latents(self, n_batch, latents, t=None):
        """
        Generate samples using manually specified latent variables.
        Args:
        - n_batch: Number of samples to generate.
        - latents: List of latent variables to use for each decoder block.
        - t: Temperature parameter for sampling (adjusts variance).
        
        Returns:
        - Samples from the model's output distribution.
        """
        # Pass the manually set latents through the decoder
        px_z = self.decoder.forward_manual_latents(n_batch, latents, t=t)
        # Use the output network to sample from the predicted distribution
        return self.decoder.out_net.sample(px_z)
import torch
import numpy as np
from vae_block.decoder import Decoder
from vae_block.encoder import Encoder
from vae_block.DmolNet import HModule


# Define the Variational Autoencoder (VAE) class, which integrates the encoder and decoder
class VAE(HModule):
    def build(self):
        # Initialize the encoder and decoder using the provided hyperparameters (H)
        self.encoder = Encoder(self.H)
        self.decoder = Decoder(self.H)

    def forward(self, x, x_target):
        ## Input:
        # ====================================================================================
        # x: Input data.
        # x_target: Target data (usually the same as `x`, but could differ for augmented data).
        ## Return:
        # ====================================================================================
        # A dictionary containing:
        # elbo: Evidence Lower Bound, the objective being optimized.
        # distortion: Reconstruction loss (negative log-likelihood of the data given the latents).
        # rate: KL divergence term, regularizing the latent space.
        # Pass the input through the encoder to get activations
        act_output = self.encoder.forward(x)
        px_z, stats = self.decoder.forward(act_output)
        # Compute the negative log-likelihood (NLL) of the reconstruction
        negativell_per_pixel = self.decoder.out_net.negative_likelihood(px_z, x_target)
        # Initialize the rate (KL divergence) term
        kl_divergnece_per_pixel = torch.zeros_like(negativell_per_pixel)
        # Compute the total number of dimensions in the input
        total_dim = 1
        for dim in x.shape[1:]:
            total_dim = total_dim * dim
        
        # Accumulate the KL divergence from all decoder blocks
        for statdict in stats:
            kl_divergnece_per_pixel += statdict['kl'].sum(dim=(1, 2, 3))  # Sum over spatial dimensions
        
        # Normalize KL divergence by the number of dimensions
        kl_divergnece_per_pixel = kl_divergnece_per_pixel/total_dim
        final_loss = (negativell_per_pixel + kl_divergnece_per_pixel).mean() # Compute the loss (elbo)
        distortion_per_pixel_mean = negativell_per_pixel.mean()
        rate_mean = kl_divergnece_per_pixel.mean()

        # Return the computed metrics
        # In this part, basically returning the ELBO (Loss function) value.
        return dict(
            elbo=final_loss,  # evidence Lower Bound
            distortion=distortion_per_pixel_mean,  # reconstruction loss
            rate=rate_mean  # KL divergence regularization term
        )


    def forward_get_latents(self, x):
        # Input:
        # ====================================================================================
        # x: Input data.
        # Return:
        # ====================================================================================
        # stats: List of KL divergence statistics and latent samples from decoder blocks.
        # Pass the input through the encoder to get activations
        # Pass activations through the decoder to get latents and their statistics
        act_output = self.encoder.forward(x)

        _, statistics = self.decoder.forward(act_output, get_latents=True)
        return statistics


    def forward_unconditional_samples(self, batch_num, temperature=None):
        # Generate unconditional samples from the prior.
        # Args:
        # - n_batch: Number of samples to generate.
        # - t: Temperature parameter for sampling (adjusts variance).
        # ====================================================================================
        # Returns:
        # Samples from the model's output distribution.
        # Generate samples from the decoder without conditioning on input data
        p_x_when_z = self.decoder.forward_unconditional(batch_num, temperature=temperature)
        return self.decoder.out_net.get_sample(p_x_when_z)


    def forward_samples_set_latents(self, batch_num, latent_variables, temperature=None):
        # Generate samples using manually specified latent variables.
        # ====================================================================================
        # Args:
        # n_batch: Number of samples to generate.
        # latents: List of latent variables to use for each decoder block.
        # t: Temperature parameter for sampling (adjusts variance).
        # ====================================================================================
        # Returns:
        # - Samples from the model's output distribution.
        # Pass the manually set latents through the decoder
        p_x_when_z = self.decoder.forward_with_manual_latent_varialbes(batch_num, latent_variables, temperature=temperature)
        return_val = self.decoder.out_net.get_sample(p_x_when_z)
        return return_val
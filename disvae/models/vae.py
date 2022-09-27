"""
Module containing the main VAE class.
"""
import random
import numpy as np 

import torch
from torch import nn, optim
from torch.nn import functional as F

from disvae.utils.initialization import weights_init
from .encoders import get_encoder
from .decoders import get_decoder
from .utilities import get_utility 

from numpy import longdouble

MODELS = ["Burgess"]
UTILTIIES = ["Malloy"]


def init_specific_model(model_type, utility_type, img_size, latent_dim):
    """Return an instance of a VAE with encoder and decoder from `model_type`."""
    model_type = model_type.lower().capitalize()
    if model_type not in MODELS:
        err = "Unkown model_type={}. Possible values: {}"
        raise ValueError(err.format(model_type, MODELS))

    encoder = get_encoder(model_type)
    decoder = get_decoder(model_type)
    utility = get_utility(utility_type)
    model = VAE(img_size, encoder, decoder, utility, latent_dim)
    model.model_type = model_type  # store to help reloading
    model.utility_type = utility_type  # store to help reloading
    return model


class VAE(nn.Module):
    def __init__(self, img_size, encoder, decoder, utility, latent_dim):
        """
        Class which defines model and forward pass.

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).
        """
        super(VAE, self).__init__()

        if list(img_size[1:]) not in [[32, 32], [64, 64]]:
            raise RuntimeError("{} sized images not supported. Only (None, 32, 32) and (None, 64, 64) supported. Build your own architecture or reshape images!".format(img_size))
        
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.num_pixels = self.img_size[1] * self.img_size[2]
        self.encoder = encoder(img_size, self.latent_dim)
        self.decoder = decoder(img_size, self.latent_dim)
        self.utility = utility(self.latent_dim)

        self.reset_parameters()

    def reparameterize(self, mean, logvar):
        """
        Samples from a normal distribution using the reparameterization trick.

        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (batch_size, latent_dim)

        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (batch_size,
            latent_dim)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + std * eps
        else:
            # Reconstruction mode
            return mean

    def forward(self, x):
        """
        Forward pass of model.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        latent_dist = self.encoder(x)
        latent_sample = self.reparameterize(*latent_dist)
        reconstruct = self.decoder(latent_sample)
        #util_input = torch.cat((latent_dist[0], latent_dist[1]), 1)
        #util_input = torch.from_numpy(np.array(self.random_sample(x))).float()
        util_input = latent_sample
        utility = self.utility(util_input)
        return reconstruct, latent_dist, latent_sample, utility 

    def reset_parameters(self):
        self.apply(weights_init)

    def sample_latent(self, x):
        """
        Returns a sample from the latent distribution.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        latent_dist = self.encoder(x)
        latent_sample = self.reparameterize(*latent_dist)
        return latent_sample
    
    def random_sample(self, x):
        """
        Returns a sample from the latent distribution without using the reparameterization trick 
        Note that this method uses pytorch and will detatch the gradient 
        Assumes a diagonal covariance matrix 

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        latent_dists = self.encoder(x)

        latent_samples = []

        for latent_means, latent_logvars in zip(latent_dists[0], latent_dists[1]):
            latent_sample = []
            for mean, logvar in zip(latent_means, latent_logvars): # independent sampling due to horizontal covariance matrix
                mean = mean.detach().numpy()
                std = np.exp(0.5 * logvar.detach().numpy())

                sample = np.random.normal(mean, std, 1)[0]
                latent_sample.append(sample)
            latent_samples.append(latent_sample)
        
        latent_samples = np.array(latent_samples)

        return latent_samples

        """
        assert(False)

        if(latent_dist[1].detach().numpy().shape[0] > 1):
            latent_dist_1 = latent_dist[1].detach().numpy().squeeze()
        else:
            latent_dist_1 = latent_dist[1].detach().numpy()
        
        covs = []
        for latent in latent_dist_1:
            latent = np.clip(latent, None, 10)
            covs.append(np.diag(np.exp(latent) ** 2))
        covs = np.array(covs)
        #covs = np.diag(np.exp(latent_dist[1].detach().numpy().squeeze()) ** 2)
        
        if(latent_dist[0].detach().numpy().shape[0] > 1):
            means = latent_dist[0].detach().numpy().squeeze()
        else:
            means = latent_dist[0].detach().numpy()
        
        latent_samples = []
        for mean, cov in zip(means, covs):
            latent_samples.append(np.random.multivariate_normal(mean, cov))
        latent_samples = np.array(latent_samples)
        """
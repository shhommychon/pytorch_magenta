"""Base Music Variational Autoencoder (MusicVAE) model."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseEncoder(nn.Module):
    def __init__(self):
        super(BaseEncoder, self).__init__()

class BaseDecoder(nn.Module):
    def __init__(self):
        super(BaseDecoder, self).__init__()

class MusicVAE(nn.Module):
    """MusicVAE

    paper: https://arxiv.org/abs/1803.05428
    original implementation:
        https://github.com/magenta/magenta/blob/main/magenta/models/music_vae/base_model.py#L126-L349
    
    """
    def __init__(self, encoder, decoder):
        super(MusicVAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, x):
        mu, sig = self.encoder(x)
        eps = torch.randn_like(sig, requires_grad=False)
        z = mu + sig * eps

        output = self.decoder(z)
        return output

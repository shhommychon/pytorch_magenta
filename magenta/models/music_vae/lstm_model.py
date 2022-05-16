"""LSTM-based encoders and decoders for MusicVAE."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from .base_model import BaseEncoder, BaseDecoder

class LstmEncoder(BaseEncoder):
    def __init__(
            self,
            input_size,
            hidden_size=2048,
            num_layers=2,
            batch_first=True,
            latent_size=512
        ):
        super(LstmEncoder, self).__init__()

        self.LSTM = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=batch_first, 
            bidirectional=False
        )

        self.mu = nn.Linear(in_features=hidden_size, out_features=latent_size)
        self.sigma = nn.Linear(in_features=hidden_size, out_features=latent_size)
    
    def forward(self, x):
        _, (h, _) = self.LSTM(x)
        h_T = rearrange(h[-1, :, :], "d b h -> b d_h")

        mu = self.mu(h_T)
        sig = torch.log(torch.exp(self.sig(h_T)) + 1)

        return mu, sig

class BidirectionalLstmEncoder(BaseEncoder):
    def __init__(
            self,
            input_size,
            hidden_size=2048,
            num_layers=2,
            batch_first=True,
            latent_size=512
        ):
        super(BidirectionalLstmEncoder, self).__init__()

        self.BiLSTM = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=batch_first, 
            bidirectional=True
        )

        self.mu = nn.Linear(in_features=2*hidden_size, out_features=latent_size)
        self.sig = nn.Linear(in_features=2*hidden_size, out_features=latent_size)
    
    def forward(self, x):
        _, (h, _) = self.BiLSTM(x)
        h_T = rearrange(h, "d_n b h -> d n b h", d=2)[:, -1, :, :]
        h_T = rearrange(h_T, "d b h -> b d_h")

        mu = self.mu(h_T)
        sig = torch.log(torch.exp(self.sig(h_T)) + 1)

        return mu, sig

class BaseLstmDecoder(BaseDecoder):
    def __init__(
            self,
            input_size,
            hidden_size=1024,
            num_layers=2,
            batch_first=True,
            proj_size=512
        ):
        super(BaseLstmDecoder, self).__init__()

        self.LSTM = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=batch_first, 
            bidirectional=False, 
            proj_size=proj_size
        )
    
    def forward(self, x):
        pass

class HierarchicalLstmDecoder(BaseDecoder):
    def __init__(
            self,
            con_input_size,
            dec_proj_size,
            batch_first=True,
            con_hidden_size=1024,
            con_num_layers=2,
            con_proj_size=512,
            dec_hidden_size=1024,
            dec_num_layers=2,
        ):
        super(HierarchicalLstmDecoder, self).__init__()

        conductor = nn.LSTM(
            input_size=con_input_size, 
            hidden_size=con_hidden_size, 
            num_layers=con_num_layers, 
            batch_first=batch_first, 
            bidirectional=False, 
            proj_size=con_proj_size
        )

        decoder = nn.LSTM(
            input_size=con_proj_size, 
            hidden_size=dec_hidden_size, 
            num_layers=dec_num_layers, 
            batch_first=batch_first, 
            bidirectional=False,
            proj_size=dec_proj_size
        )
    
    def forward(self, x):
        pass

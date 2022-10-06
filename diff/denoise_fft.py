import torch
from torch.nn import functional as F
import torch.nn as nn
import math

if __package__ is None or __package__ == '' or __package__ == 'diff':
    import hparams as hp
    from transformer.Models import Decoder
else:
    from .. import hparams as hp
    from ..transformer.Models import Decoder


Linear = nn.Linear

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class FFT(Decoder):
    def __init__(self, 
                 len_max_seq=hp.max_seq_len,
                 n_layers=hp.diff_decoder_n_layer,
                 n_head=hp.diff_decoder_head,
                 d_k=hp.decoder_dim // hp.diff_decoder_head,
                 d_v=hp.decoder_dim // hp.diff_decoder_head,
                 d_model=hp.decoder_dim,
                 d_inner=hp.diff_decoder_conv1d_filter_size,
                 dropout=hp.diff_dropout):
        super().__init__(len_max_seq, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout)
        dim = d_model
        self.input_projection = Conv1d(hp.num_mels, dim, 1)
        self.diffusion_embedding = SinusoidalPosEmb(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim)
        )
        self.get_mel_out = Linear(dim, 80, bias=True)
        self.get_decode_inp = Linear(dim + dim + dim,
                                     dim)

    def forward(self, spec, diffusion_step, cond, mel_pos):
        """
        :param spec: [B, 1, 80, T]
        :param diffusion_step: [B, 1]
        :param cond: [B, M, T]
        :param mel_pos: [B, T]
        :return:
        """
        x = spec[:, 0]
        x = self.input_projection(x).permute([0, 2, 1])  #  [B, T, residual_channel]
        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step)  # [B, dim]
        cond = cond.permute([0, 2, 1])  # [B, T, M]

        seq_len = cond.shape[1]  # [T_mel]
        time_embed = diffusion_step[:, None, :]  # [B, 1, dim]
        time_embed = time_embed.repeat([1, seq_len, 1])  # # [B, T, dim]

        decoder_inp = torch.cat([x, cond, time_embed], dim=-1)  # [B, T, dim + H + dim]
        decoder_inp = self.get_decode_inp(decoder_inp)  # [B, T, H]
        x = decoder_inp

        x = Decoder.forward(self, x, mel_pos) # [B, T, H]
        x = self.get_mel_out(x).permute([0, 2, 1]) # [B, 80, T]
        return x[:, None, :, :]

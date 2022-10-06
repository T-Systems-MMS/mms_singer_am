# Adapted from MoonInTheRiver/DiffSinger

import math
import random
from functools import partial
from inspect import isfunction
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

if __package__ is None or __package__ == '' or __package__ == 'diff':
    import hparams as hp
    from .denoise_wavenet import DiffNet
    from .denoise_fft import FFT
else:
    from .. import hparams as hp
    from .denoise_wavenet import DiffNet
    from .denoise_fft import FFT


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def linear_beta_schedule(timesteps, max_beta=0.01):
    """
    linear schedule
    """
    betas = np.linspace(1e-4, max_beta, timesteps)
    return betas


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)


beta_schedule = {
    "cosine": cosine_beta_schedule,
    "linear": linear_beta_schedule,
}

class GaussianDiffusion(nn.Module):
    def __init__(self, denoise_fn=hp.denoise_fn, timesteps=hp.diff_timesteps, K_step=hp.diff_K_step, loss_type=hp.diff_loss_type, betas=None, spec_min=None, spec_max=None):
        super().__init__()
        if denoise_fn == 'fft':
          self.denoise_fn = FFT()
        elif denoise_fn == 'wavenet':
          self.denoise_fn = DiffNet()
        else:
          self.denoise_fn = denoise_fn
          
        self.mel_bins = hp.num_mels

        if exists(betas):
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            betas = beta_schedule[hp.beta_schedule_type](timesteps)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.K_step = K_step
        self.loss_type = loss_type

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if spec_min is None:
            spec_min = np.ones(hp.num_mels) * -0.5
        if spec_max is None:
            spec_max = np.ones(hp.num_mels) * 0.5
        self.register_buffer('spec_min', torch.FloatTensor(spec_min)[None, None, :hp.num_mels])
        self.register_buffer('spec_max', torch.FloatTensor(spec_max)[None, None, :hp.num_mels])

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, cond, mel_pos, clip_denoised: bool):
        noise_pred = self.denoise_fn(x, t, cond=cond, mel_pos=mel_pos)
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise_pred)

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, cond, mel_pos, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, cond=cond, mel_pos=mel_pos, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, cond, mel_pos, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.denoise_fn(x_noisy, t, cond, mel_pos)

        nonpadding = (mel_pos != 0).float()

        if self.loss_type == 'l1':
            loss = ((noise - x_recon).abs() * nonpadding.unsqueeze(1)).mean()

        elif self.loss_type == 'l2':
            loss = ((noise - x_recon) ** 2 * nonpadding.unsqueeze(1)).mean()
        else:
            raise NotImplementedError()

        return loss

    def forward(self, fs_decoder_input, fs_mels=None, mel_targets=None, mel_pos=None, infer=True):
        if infer:
            assert hp.gaussian_start or fs_mels is not None
            assert mel_pos is not None
        else:
            assert mel_targets is not None
            assert mel_pos is not None

        b, *_, device = *fs_decoder_input.shape, fs_decoder_input.device

        cond = fs_decoder_input.transpose(1, 2)
        ret = {}

        if not infer:
            t = torch.randint(0, self.K_step, (b,), device=device).long()
            x = mel_targets
            x = self.norm_spec(x)
            x = x.transpose(1, 2)[:, None, :, :]  # [B, 1, M, T]
            ret['diff_loss'] = self.p_losses(x, t, cond, mel_pos)
        else:
            t = self.K_step
        
            if hp.gaussian_start:
                shape = (cond.shape[0], 1, self.mel_bins, cond.shape[2])
                x = torch.randn(shape, device=device)
            else:
                fs_mels = self.norm_spec(fs_mels)
                fs_mels = fs_mels.transpose(1, 2)[:, None, :, :]
                x = self.q_sample(x_start=fs_mels, t=torch.tensor([t - 1], device=device).long())
            
            for i in reversed(range(0, t)):
                x = self.p_sample(x, torch.full((b,), i, device=device, dtype=torch.long), cond, mel_pos)
            x = x[:, 0].transpose(1, 2)
            ret['mel_out'] = self.denorm_spec(x)
        return ret

    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min


# Apply noise to a batch of mel specs according to current hyperparameters
# Mel must be normalized

alphas_cumprod = np.cumprod(1. - beta_schedule[hp.beta_schedule_type](hp.diff_timesteps), axis=0)
def apply_noise(mel, t):
    assert mel.min() >= -1
    assert mel.max() <= 1
    assert len(t) == mel.shape[0]
    assert torch.all(t >= 0)
    assert torch.all(t <= hp.diff_timesteps)

    sqrt_alphas_cumprod = torch.tensor(np.sqrt(alphas_cumprod), device=mel.device)
    sqrt_one_minus_alphas_cumprod = torch.tensor(np.sqrt(1. - alphas_cumprod), device=mel.device)
    
    noise = torch.randn_like(mel)
    return (
        extract(sqrt_alphas_cumprod, t, mel.shape) * mel +
        extract(sqrt_one_minus_alphas_cumprod, t, mel.shape) * noise
    )

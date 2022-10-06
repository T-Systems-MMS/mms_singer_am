import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from numba import jit
import numpy as np
import copy
import math
import warnings
from typing import Optional

if __package__ is None or __package__ == '':
    import hparams as hp
    import utils
else:
    from . import hparams as hp
    from . import utils


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i)
                               for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# An alternative to create_alignment that can run on the gpu
def create_alignment_torch(duration_predictor_output):
    max_mel_len = duration_predictor_output.sum(-1).max()
    batch_dim = duration_predictor_output.shape[0]
    # Zeros at the bottom where the cumsum is smaller than arange
    temp_align = duration_predictor_output.cumsum(1).unsqueeze(1).repeat_interleave(max_mel_len, 1) - torch.arange(max_mel_len, device=duration_predictor_output.device).repeat(batch_dim,1).unsqueeze(-1)
    # Zeros at the top where the durations are smaller than their cumsum (extracted from temp_align)
    temp_align_2 = (duration_predictor_output.unsqueeze(1).repeat_interleave(max_mel_len, 1) - (temp_align * (temp_align > 0)) >= 0)

    return ((temp_align > 0) * temp_align_2).float()

# Calculate the error the duration predictor made on each syllable and distribute it to phoneme-level
def syllable_guidance_error(duration_predictor_output, syllable_dur, syllable_pos):
    # Calculate a syllable alignment matrix that aligns from phoneme dimension to syllable dimension
    max_pos = syllable_pos.max()
    syllable_mat = syllable_pos.unsqueeze(-1).repeat_interleave(max_pos, dim=2) - torch.arange(max_pos, device=syllable_pos.device)
    syllable_mat[syllable_mat != 1] = 0
    syllable_mat = syllable_mat.float()
    # Calc the error for each syllable by summing from phoneme to syllable dimension (first @) and then expanding back to phonemes (second @)
    # The second bracket then holds the duration sums for each syllable as in syllable_dur but from the duration predictor
    # Then subtract the ground truth durations from the predictions to get an error term
    syllable_duration_error = syllable_dur.float() - (duration_predictor_output.detach().unsqueeze(1) @ syllable_mat @ syllable_mat.transpose(1,2)).squeeze(1)
    
    # Distribute relative to phoneme length, meaning if a phoneme makes up 5% of the syllable duration according to the dp, it will only take 5% of the error
    syllable_mat *= duration_predictor_output.unsqueeze(-1)
    syllable_mat /= syllable_mat.sum(axis=1, keepdim=True)
    syllable_mat = syllable_mat.sum(axis=2)
    
    syllable_duration_error *= syllable_mat

    return syllable_duration_error


class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self):
        super(LengthRegulator, self).__init__()
        self.duration_predictor = DurationPredictor()

    def LR(self, x, duration_predictor_output, mel_max_length:Optional[int]):
        #expand_length = torch.max(
        #    torch.sum(duration_predictor_output, -1), -1)[0]
        
        #alignment = torch.zeros(duration_predictor_output.size(0),
        #                        expand_length,
        #                        duration_predictor_output.size(1)).numpy()
        #alignment = create_alignment(alignment,
        #                             duration_predictor_output.cpu().numpy())
        #alignment = torch.from_numpy(alignment).detach().to(x.device)
        alignment = create_alignment_torch(duration_predictor_output.detach())

        output = alignment @ x
        if mel_max_length is not None:
            output = F.pad(
                output, [0, 0, 0, mel_max_length-output.size(1), 0, 0])
            alignment = F.pad(
                alignment, [0, 0, 0, mel_max_length-alignment.size(1), 0, 0])
        if hp.limit_seq_len and output.size(1) >= hp.max_seq_len:
            warnings.warn(f'Duration predictor predicted longer sequence than decoder can parse: {output.size(1)}')
            output = F.pad(
                output, [0, 0, 0, hp.max_seq_len-output.size(1), 0, 0])
            alignment = F.pad(
                alignment, [0, 0, 0, hp.max_seq_len-alignment.size(1), 0, 0])
        return output, alignment
    

    def forward(self, x, alpha:float=1.0, target=None, mel_max_length=None, syllable_dur=None, syllable_pos=None):
        duration_predictor_output = self.duration_predictor(x)

        if target is not None:
            output, alignment = self.LR(x, target, mel_max_length=mel_max_length)
            return output, alignment, duration_predictor_output
        else:
            if syllable_pos is not None:
                duration_predictor_output[syllable_pos == 0] = 0

            if syllable_dur is not None and syllable_pos is not None:
                
                duration_predictor_output += syllable_guidance_error(duration_predictor_output, syllable_dur, syllable_pos)
            
            duration_predictor_output = (
                (duration_predictor_output + 0.5) * alpha).int()

            output, alignment = self.LR(x, duration_predictor_output, mel_max_length=None)
            #mel_pos = torch.stack(
            #    [torch.Tensor([i+1 for i in range(output.size(1))])]).long().to(x.device)
            mel_pos = torch.arange(output.shape[1], dtype=torch.long, device=x.device, requires_grad=False).unsqueeze(0)+1
            mel_pos = mel_pos.repeat(output.shape[0], 1)
            for batch_idx in range(output.shape[0]):
                mel_pos[batch_idx, duration_predictor_output[batch_idx].detach().sum().cpu():]=0
            return output, alignment, duration_predictor_output, mel_pos


class DurationPredictor(nn.Module):
    """ Duration Predictor """

    def __init__(self):
        super(DurationPredictor, self).__init__()

        self.input_size = hp.encoder_dim
        self.filter_size = hp.duration_predictor_filter_size
        self.kernel = hp.duration_predictor_kernel_size
        self.conv_output_size = hp.duration_predictor_filter_size
        self.dropout = hp.dropout

        self.conv_layer = nn.Sequential(OrderedDict([
            ("conv1d_1", Conv(self.input_size,
                              self.filter_size,
                              kernel_size=self.kernel,
                              padding=1)),
            ("layer_norm_1", nn.LayerNorm(self.filter_size)),
            ("relu_1", nn.ReLU()),
            ("dropout_1", nn.Dropout(self.dropout)),
            ("conv1d_2", Conv(self.filter_size,
                              self.filter_size,
                              kernel_size=self.kernel,
                              padding=1)),
            ("layer_norm_2", nn.LayerNorm(self.filter_size)),
            ("relu_2", nn.ReLU()),
            ("dropout_2", nn.Dropout(self.dropout))
        ]))

        self.linear_layer = Linear(self.conv_output_size, 1)

    def forward(self, encoder_output):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.exp()
        out = out.squeeze(2)
        return out


class BatchNormConv1d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, padding,
                 activation=None, w_init_gain='linear'):
        super(BatchNormConv1d, self).__init__()
        self.conv1d = nn.Conv1d(in_dim, out_dim,
                                kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_dim)
        self.activation = activation

        torch.nn.init.xavier_uniform_(
            self.conv1d.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        x = self.conv1d(x)
        if self.activation is not None:
            x = self.activation(x)
        return self.bn(x)


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 w_init='linear'):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=bias)

        nn.init.xavier_uniform_(
            self.conv.weight, gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x


class Linear(nn.Module):
    """
    Linear Module
    """

    def __init__(self, in_dim, out_dim, bias=True, w_init='linear'):
        """
        :param in_dim: dimension of input
        :param out_dim: dimension of output
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        return self.linear_layer(x)


class Highway(nn.Module):
    def __init__(self, in_size, out_size):
        super(Highway, self).__init__()
        self.H = nn.Linear(in_size, out_size)
        self.H.bias.data.zero_()
        self.T = nn.Linear(in_size, out_size)
        self.T.bias.data.fill_(-1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        H = self.relu(self.H(inputs))
        T = self.sigmoid(self.T(inputs))
        return H * T + inputs * (1.0 - T)


class Prenet(nn.Module):
    """
    Prenet before passing through the network
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(Prenet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.layer = nn.Sequential(OrderedDict([
            ('fc1', Linear(self.input_size, self.hidden_size)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(0.5)),
            ('fc2', Linear(self.hidden_size, self.output_size)),
            ('relu2', nn.ReLU()),
            ('dropout2', nn.Dropout(0.5)),
        ]))

    def forward(self, x):
        out = self.layer(x)
        return out


class CBHG(nn.Module):
    """CBHG module: a recurrent neural network composed of:
        - 1-d convolution banks
        - Highway networks + residual connections
        - Bidirectional gated recurrent units
    """

    def __init__(self, in_dim, K=16, projections=[128, 128]):
        super(CBHG, self).__init__()
        self.in_dim = in_dim
        self.relu = nn.ReLU()
        self.conv1d_banks = nn.ModuleList(
            [BatchNormConv1d(in_dim, in_dim, kernel_size=k, stride=1,
                             padding=k // 2, activation=self.relu)
             for k in range(1, K + 1)])
        self.max_pool1d = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

        in_sizes = [K * in_dim] + projections[:-1]
        activations = [self.relu] * (len(projections) - 1) + [None]
        self.conv1d_projections = nn.ModuleList(
            [BatchNormConv1d(in_size, out_size, kernel_size=3, stride=1,
                             padding=1, activation=ac)
             for (in_size, out_size, ac) in zip(
                 in_sizes, projections, activations)])

        self.pre_highway = nn.Linear(projections[-1], in_dim, bias=False)
        self.highways = nn.ModuleList(
            [Highway(in_dim, in_dim) for _ in range(4)])

        self.gru = nn.GRU(
            in_dim, in_dim, 1, batch_first=True, bidirectional=True)

    def forward(self, inputs, input_lengths=None):
        # (B, T_in, in_dim)
        x = inputs

        # Needed to perform conv1d on time-axis
        # (B, in_dim, T_in)
        if x.size(-1) == self.in_dim:
            x = x.transpose(1, 2)

        T = x.size(-1)

        # (B, in_dim*K, T_in)
        # Concat conv1d bank outputs
        x = torch.cat([conv1d(x)[:, :, :T]
                       for conv1d in self.conv1d_banks], dim=1)
        assert x.size(1) == self.in_dim * len(self.conv1d_banks)
        x = self.max_pool1d(x)[:, :, :T]

        for conv1d in self.conv1d_projections:
            x = conv1d(x)

        # (B, T_in, in_dim)
        # Back to the original shape
        x = x.transpose(1, 2)

        if x.size(-1) != self.in_dim:
            x = self.pre_highway(x)

        # Residual connection
        x += inputs
        for highway in self.highways:
            x = highway(x)

        if input_lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, input_lengths, batch_first=True)

        # (B, T_in, in_dim*2)
        #self.gru.flatten_parameters()
        outputs, _ = self.gru(x)

        if input_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(
                outputs, batch_first=True)

        return outputs


if __name__ == "__main__":
    # TEST
    a = torch.Tensor([[2, 3, 4], [1, 2, 3]])
    b = torch.Tensor([[5, 6, 7], [7, 8, 9]])
    c = torch.stack([a, b])

    d = torch.Tensor([[1, 4], [6, 3]]).int()
    expand_max_len = torch.max(torch.sum(d, -1), -1)[0]
    base = torch.zeros(c.size(0), expand_max_len, c.size(1))

    alignment = create_alignment_torch(d)
    print(alignment)
    print(alignment @ c)

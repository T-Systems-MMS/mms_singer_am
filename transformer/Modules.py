import torch
import torch.nn as nn
import numpy as np

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, input_mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if input_mask is not None:
            attn = attn.masked_fill(~input_mask.unsqueeze(1).expand(-1, input_mask.shape[1], -1), -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output


if __name__ == '__main__':
    bs = 3
    seql = 150
    q = torch.randn(bs, seql, 256)
    k = torch.randn(bs, seql, 256)
    v = torch.randn(bs, seql, 256)
    mask = q[:,:,0] < -1

    attn = ScaledDotProductAttention(1)
    out, _ = attn(q, k, v, mask)

    attn = LocalAttention()
    out2, _ = attn(q, k, v, mask)

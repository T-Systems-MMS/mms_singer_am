import torch
import torch.nn as nn
import numpy as np

if __package__ is None or __package__ == '' or __package__ == 'transformer':
    import hparams as hp
    import transformer.Constants as Constants
    from transformer.Layers import FFTBlock, PreNet, PostNet, Linear
else:
    from .. import hparams as hp
    from . import Constants as Constants
    from .Layers import FFTBlock, PreNet, PostNet, Linear

def get_non_pad_mask(seq):
    assert seq.dim() == 2, f'Seq dim {seq.dim()} cannot be used as mask'
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


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


def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.ne(Constants.PAD)
    #padding_mask = padding_mask.unsqueeze(
    #    1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


class Encoder(nn.Module):
    ''' Encoder '''

    def __init__(self,
                 n_src_vocab=hp.vocab_size,
                 n_note_vocab=hp.note_vocab_size,
                 n_duration_vocab=hp.syllable_duration_vocab_size,
                 len_max_seq=hp.max_seq_len_txt,
                 n_phoneme_pos=hp.phoneme_pos_vocab_size,
                 n_syllable_pos=hp.syllable_pos_vocab_size,
                 d_word_vec=hp.phoneme_emb_dim,
                 d_syl_vec=hp.syllable_emb_dim,
                 n_layers=hp.encoder_n_layer,
                 n_head=hp.encoder_head,
                 d_k=hp.encoder_dim // hp.encoder_head,
                 d_v=hp.encoder_dim // hp.encoder_head,
                 d_model=hp.encoder_dim,
                 d_inner=hp.encoder_conv1d_filter_size,
                 dropout=hp.dropout,
                 attn_type=hp.encoder_attn_type,
                 attn_local_context=hp.encoder_attn_local_context):

        super(Encoder, self).__init__()

        self.src_word_emb = nn.Embedding(n_src_vocab,
                                         d_word_vec,
                                         padding_idx=Constants.PAD)

        self.note_emb = nn.Embedding(n_note_vocab,
                                     d_syl_vec,
                                     padding_idx=Constants.PAD)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_phoneme_pos, d_word_vec, padding_idx=0),
            freeze=True)

        self.syllable_duration_emb = nn.Embedding(n_duration_vocab,
                                               d_syl_vec,
                                               padding_idx=Constants.PAD)
        
        self.syllable_pos_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_syllable_pos, d_syl_vec, padding_idx=0),
            freeze=True)


        self.layer_stack = nn.ModuleList([FFTBlock(
            d_model, d_inner, n_head, d_k, d_v, dropout=dropout, attn_type=attn_type, attn_local_context=attn_local_context, attn_local_pos=False) for _ in range(n_layers)])

    def forward(self, src_seq, note_seq, syllable_dur_seq, syllable_pos, src_pos):
        assert src_seq.shape[1] > 0, 'Sequence length zero'

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_pos)

        # -- Forward
        enc_output = self.src_word_emb(src_seq)
        enc_output += self.position_enc(src_pos)

        syllable_output = self.syllable_duration_emb(syllable_dur_seq)
        syllable_output += self.note_emb(note_seq)
        syllable_output += self.syllable_pos_enc(syllable_pos)
        
        # Position embeddings are not mixed together but kept separate
        enc_output = torch.cat((enc_output, syllable_output), dim=2)

        for enc_layer in self.layer_stack:
            enc_output = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)

        return enc_output, non_pad_mask


class Decoder(nn.Module):
    """ Decoder """

    def __init__(self,
                 len_max_seq=hp.max_seq_len,
                 n_layers=hp.decoder_n_layer,
                 n_head=hp.decoder_head,
                 d_k=hp.decoder_dim // hp.decoder_head,
                 d_v=hp.decoder_dim // hp.decoder_head,
                 d_model=hp.decoder_dim,
                 d_inner=hp.decoder_conv1d_filter_size,
                 dropout=hp.dropout,
                 attn_type=hp.decoder_attn_type,
                 attn_local_context=hp.decoder_attn_local_context,
                 attn_local_pos=hp.decoder_attn_local_pos):

        super(Decoder, self).__init__()

        n_position = len_max_seq + 1
        self.max_seq_len = len_max_seq
        self.d_model = d_model

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_model, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([FFTBlock(
            d_model, d_inner, n_head, d_k, d_v, dropout=dropout, attn_type=attn_type, attn_local_context=attn_local_context, attn_local_pos=attn_local_pos) for _ in range(n_layers)])

    def forward(self, enc_seq, enc_pos):
        assert enc_seq.shape[1] > 0, 'Sequence length zero'

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=enc_pos, seq_q=enc_pos)
        non_pad_mask = get_non_pad_mask(enc_pos)

        # -- Forward
        if not hp.decoder_attn_local_pos:
            if not hp.limit_seq_len and enc_seq.shape[1] > self.max_seq_len:
                # -- Prepare masks
                enc_seq = enc_seq + get_sinusoid_encoding_table(enc_seq.shape[1], self.d_model, padding_idx=0)[: enc_seq.shape[1], :].unsqueeze(0).expand(enc_seq.shape[0], -1, -1).to(enc_seq.device)
            else:
                enc_seq = enc_seq + self.position_enc(enc_pos)

        for dec_layer in self.layer_stack:
            enc_seq = dec_layer(
                enc_seq,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)

        return enc_seq

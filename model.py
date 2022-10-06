import torch
import torch.nn as nn
import torch.nn.functional as F

if __package__ is None or __package__ == '':
    import hparams as hp
    import utils
    import audio
    from gst import ReferenceEncoder, StyleTokenLayer
    from transformer.Models import Encoder, Decoder
    from transformer.Layers import Linear, PostNet
    from modules import LengthRegulator, CBHG
else:
    from . import hparams as hp
    from . import utils
    from . import audio
    from .gst import ReferenceEncoder, StyleTokenLayer
    from .transformer.Models import Encoder, Decoder
    from .transformer.Layers import Linear, PostNet
    from .modules import LengthRegulator, CBHG

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FastSpeech(nn.Module):
    """ FastSpeech """

    def __init__(self):
        super(FastSpeech, self).__init__()

        self.encoder = Encoder()
        self.length_regulator = LengthRegulator()
        self.decoder = Decoder()
        if hp.use_gst:
            self.reference_encoder = ReferenceEncoder()
            self.style_token_layer = StyleTokenLayer()

        self.mel_linear = Linear(hp.decoder_dim, hp.num_mels+2)
        self.postnet = CBHG(hp.num_mels, K=8,
                            projections=[256, hp.num_mels])
        self.last_linear = Linear(hp.num_mels * 2, hp.num_mels)
        self.note_conversion = torch.nn.parameter.Parameter(torch.zeros(hp.note_vocab_size+1), requires_grad=False)

    def set_note_conversion(self, note_conv):
        with torch.no_grad():
            assert len(note_conv) >= len(self.note_conversion), 'Note conversion vocab is smaller than hparams.note_vocab_size'
            note_conv = torch.tensor(note_conv, device=self.note_conversion.device)
            self.note_conversion.copy_(note_conv[0:len(self.note_conversion)])
    
    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~utils.get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)

    def forward(self, src_seq, note_seq, syllable_dur_seq, syllable_pos, src_pos, mel_pos=None, mel_max_length=None, length_target=None, voiced_target=None, alpha=1.0, syllable_dur_guidance=None, stl_target=None, stl_weights=None):
        
        encoder_output, _ = self.encoder(src_seq, note_seq, syllable_dur_seq, syllable_pos, src_pos)

        # Use GST if wanted
        if stl_target is not None:
            encoder_output += self.style_token_layer(self.reference_encoder(stl_target))
        elif stl_weights is not None:
            encoder_output += self.style_token_layer.weighted_style(stl_weights)
        

        if length_target is not None:
            assert mel_max_length is not None

            length_regulator_output, alignment, duration_predictor_output = self.length_regulator(encoder_output,
                                                                                       target=length_target,
                                                                                       alpha=alpha,
                                                                                       mel_max_length=mel_max_length)
        else:
            length_regulator_output, alignment, duration_predictor_output, mel_pos = self.length_regulator(encoder_output,
                                                                                                alpha=alpha,
                                                                                                syllable_dur=syllable_dur_guidance,
                                                                                                syllable_pos=None if syllable_dur_guidance is None else syllable_pos)
            mel_max_len = mel_pos.shape[1]

        decoder_output = self.decoder(length_regulator_output, mel_pos)

        linear_output = self.mel_linear(decoder_output)
        linear_output = self.mask_tensor(linear_output, mel_pos, mel_max_length)
        
        # Destructure into mel, f0, v/uv
        mel_output, f0_output, voiced_output = (linear_output[:,:,0:hp.num_mels], linear_output[:,:,hp.num_mels], linear_output[:,:,hp.num_mels+1])

        # Offset f0 by the note value so only the difference is predicted
        # Convert notes to hz
        notes_expanded = self.note_conversion[note_seq]
        
        # Expand notes to mel dimension
        notes_expanded = (alignment @ notes_expanded.unsqueeze(-1)).squeeze(-1)
        f0_output = (f0_output + notes_expanded) 
        
        residual = self.postnet(mel_output)
        residual = self.last_linear(residual)
        mel_postnet_output = mel_output + residual
        mel_postnet_output = self.mask_tensor(mel_postnet_output,
                                              mel_pos,
                                              mel_max_length)

        return mel_output, mel_postnet_output, mel_pos, f0_output, voiced_output, duration_predictor_output, length_regulator_output


if __name__ == "__main__":
    # Test
    model = FastSpeech()

    print(sum(param.numel() for param in model.parameters()))

    import audio
    model.set_note_conversion(audio.load_note_conversion_table('data/CSD_processed/note_conversion.csv'))

    import dataset
    #ds = dataset.BufferDataset(dataset.get_data_to_buffer(run_asserts=False))
    #dl = dataset.DataLoader(ds, batch_size=hp.batch_size * hp.batch_expand_size, collate_fn=dataset.collate_fn_tensor, shuffle=False)

    #batch = next(iter(dl))[0]
    #output = model(batch['text'], batch['note'], batch['energy'], batch['duration_guess'], batch['src_pos'])


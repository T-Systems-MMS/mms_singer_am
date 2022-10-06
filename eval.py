import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import torch
import torch.nn as nn
import argparse
import numpy as np
import random
import time
import shutil
import os
import soundfile as sf
import pickle
from PIL import Image
from tqdm import tqdm

if __package__ is None or __package__ == '':
    import audio
    import hparams as hp
    import utils
    import dataset
    import model as M
    from diff import shallow_diffusion as diff
    from data import textnotegen
    from audio import denorm_mel, mel_to_audio
else:
    from . import audio
    from . import hparams as hp
    from . import utils
    from . import dataset
    from . import model as M
    from .diff import shallow_diffusion as diff
    from .data import textnotegen
    from .audio import denorm_mel, mel_to_audio


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

class WrappedModel(nn.Module):
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module # that I actually define.
    def forward(self, x):
        return self.module(x)

def get_DNN(checkpoint_path):
    model = M.FastSpeech()
    model = WrappedModel(model)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu')['model'], strict=True)
    model = model.module
    model.to(device)
    model.eval()
    return model

def get_diff_model(checkpoint_path):
    model = diff.GaussianDiffusion()
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu')['model'], strict=True)
    model.to(device)
    model.eval()
    return model


def synthesis(model, data, alpha=1.0, guidance="none", style_token_weights=None, style_token_targets=None, choir_mode=None, choir_mode_variance=0.5, diffusion_model=None):
    text = torch.from_numpy(data['text']).long().unsqueeze(0).to(device)
    note = torch.from_numpy(data['note']).long().unsqueeze(0).to(device)
    syllable_duration = torch.from_numpy(data['syllable_duration']).long().unsqueeze(0).to(device)
    syllable_duration_raw = torch.from_numpy(data['syllable_duration_raw']).long().unsqueeze(0).to(device)
    syllable_pos = torch.from_numpy(data['syllable_pos']).long().unsqueeze(0).to(device)
    energy = torch.from_numpy(data['energy']).unsqueeze(0).to(device)

    if guidance == "phoneme":
        assert 'duration' in data, "No phoneme guidance information found, make sure the dataset includes phoneme duration labels"
        duration = torch.from_numpy(data['duration']).unsqueeze(0).to(device)
        mel_pos = torch.arange(sum(data['duration'])).long().unsqueeze(0).to(device) + 1

    
    if style_token_weights is not None:
        style_token_weights = torch.tensor(style_token_weights).unsqueeze(0).to(device)
    if style_token_targets is not None:
        assert style_token_weights is None, 'Cannot use both stl targets and stl weights'
        assert choir_mode is None, 'Cannot use choir mode with stl targets'
        style_token_targets = torch.tensor(style_token_targets).unsqueeze(0).to(device)

    src_pos = torch.tensor([i+1 for i in range(text.shape[1])]).long().unsqueeze(0).to(device)

    if choir_mode is not None:
        if style_token_weights is None:
            style_token_weights = torch.randn(hp.gst_token_num).unsqueeze(0).to(device).softmax(dim=1)
        noise_factor = choir_mode_variance
        style_token_weights = (style_token_weights.repeat(choir_mode, 1) + noise_factor * torch.randn(choir_mode, hp.gst_token_num, device=device)).softmax(dim=1)

        text = text.repeat(choir_mode, 1)
        note = note.repeat(choir_mode, 1)
        syllable_duration = syllable_duration.repeat(choir_mode, 1)
        syllable_duration_raw = syllable_duration_raw.repeat(choir_mode, 1)
        syllable_pos = syllable_pos.repeat(choir_mode, 1)
        src_pos = src_pos.repeat(choir_mode, 1)

        if guidance == 'phoneme':
            duration = duration.repeat(choir_mode, 1)
            mel_pos = mel_pos.repeat(choir_mode, 1)

    with torch.no_grad():
        _, mel, _, f0, voiced, durations, dec_input = model(text, 
                                                 note, 
                                                 syllable_duration, 
                                                 syllable_pos, 
                                                 src_pos, 
                                                 alpha=alpha, 
                                                 mel_pos=mel_pos if guidance=="phoneme" else None,
                                                 length_target=duration if guidance=="phoneme" else None,
                                                 mel_max_length=duration.sum().item() if guidance=="phoneme" else None,
                                                 syllable_dur_guidance=syllable_duration_raw if guidance == "syllable" else None,
                                                 stl_weights=style_token_weights,
                                                 stl_target=style_token_targets)
        if diffusion_model is not None:
            if guidance != 'phoneme':
                mel_pos = (torch.arange(mel.shape[1], device=device)+1).unsqueeze(0).repeat(mel.shape[0], 1)
            mel = diffusion_model(dec_input, fs_mels=mel, mel_pos=mel_pos, infer=True)['mel_out']    

    return mel.cpu().numpy(), f0.cpu().numpy(), voiced.cpu().numpy(), durations.cpu().numpy(), dec_input.cpu().numpy()

def get_data_textnotegen(ds_dir, split_mode='auto'):
    if os.path.exists(os.path.join(ds_dir, 'melody.mid')):
        data = [('textnotegen', textnotegen.load_textnotegen_format(ds_dir, split_mode))]
    else:
        data = [(f, textnotegen.load_textnotegen_format(os.path.join(ds_dir, f), split_mode)) for f in os.listdir(ds_dir) if os.path.exists(os.path.join(ds_dir, f, 'melody.mid'))]

    # Flatten it out
    data = [(f'{f}-split{i}', d) for f, da in data for i, d in enumerate(da)]
    
    data = [(f, textnotegen.fix_data(d)) for (f,d) in data]

    # Add mel targets
    data = [(f, d, None) for (f, d) in data]

    return data

def load_mel(filename):
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)['mel']
    except:
        return None

def pitch_shift(data, semitones):
    note = data['note'] + semitones
    note[data['note'] == 1] = 1
    note[note <= 1] = 1
    note[note >= hp.note_vocab_size] = hp.note_vocab_size -1
    data['note'] = note
    return data


with open(hp.phoneme_dict, 'rb') as f:
    phoneme_dict = pickle.load(f)
vowels = [v for (k,v) in phoneme_dict.items() if "1" in k]

def split_long_notes(data, max_len, vowels_only=True):
    max_len_orig = max_len
    max_len = max_len * (hp.sampling_rate / hp.hop_length)
    for idx in range(len(data['text'])):
        if data['syllable_duration_raw'][idx] > max_len and (not vowels_only or data['text'][idx] in vowels):
            data['text'] = np.insert(data['text'], idx, data['text'][idx])
            data['note'] = np.insert(data['note'], idx, data['note'][idx])
            data['syllable_duration_raw'] = np.insert(data['syllable_duration_raw'], idx, data['syllable_duration_raw'][idx])
            data['syllable_duration'] = np.insert(data['syllable_duration'], idx, data['syllable_duration'][idx])
            data['syllable_pos'] = np.insert(data['syllable_pos'], idx, data['syllable_pos'][idx])
            starting_pos = data['syllable_pos'][idx]
            # halve all syllable durations before that idx
            # halve all syllable durations after that idx and increase syllable pos
            for idx2 in range(len(data['text'])):
                if data['syllable_pos'][idx2] == starting_pos:
                    data['syllable_duration_raw'][idx2] = data['syllable_duration_raw'][idx2]//2
                    data['syllable_duration'][idx2] = data['syllable_duration'][idx2]//2
                if idx2 > idx:
                    data['syllable_pos'][idx2] = data['syllable_pos'][idx2] + 1
            return split_long_notes(data, max_len_orig, vowels_only)
    
    # If so far only vowels have been scanned, repeat again with all to make sure no notes are left
    if vowels_only:
        return split_long_notes(data, max_len_orig, False)
    return data

def pad_nosounds(data, nosound_len):
    nosound_len = int(nosound_len * (hp.sampling_rate / hp.hop_length) + 0.5)
    nosound_len_binned = audio.bin_duration(np.array([nosound_len])).item() if hp.bin_durations else nosound_len
    
    # Pre- and append nosounds for all data parts
    data['text'] = np.concatenate([[phoneme_dict['<nosound>']], data['text'], [phoneme_dict['<nosound>']]])
    data['note'] = np.concatenate([[1], data['note'], [1]])
    data['syllable_duration_raw'] = np.concatenate([[nosound_len], data['syllable_duration_raw'], [nosound_len]])
    data['syllable_duration'] = np.concatenate([[nosound_len_binned], data['syllable_duration'], [nosound_len_binned]])
    data['syllable_pos'] = np.concatenate([[1], data['syllable_pos']+1, [data['syllable_pos'].max()+2]])
    data['syllables'] = "<nosound> " + data['syllables'] + " <nosound>"

    return data

def get_data(test_ds_dir):
    files = os.listdir(test_ds_dir)
    test_data = [(os.path.splitext(os.path.basename(f))[0], 
                  dataset.load_labels(os.path.join(test_ds_dir, f)),
                  load_mel(os.path.join(test_ds_dir, '..', 'mel', f'csd-mel-{os.path.splitext(os.path.basename(f))[0]}.pkl'))) for f in files]

    
    return test_data

def process(dataset, model, alpha, result_dir, guidance='auto', use_gl=False, style_token_weights=None, use_stl_target=False, save_norm_mel=False, choir_mode=None, choir_mode_variance=0.5, diffusion_model=None):
    for filename, data, gt_mel in tqdm(dataset):
        assert not use_stl_target or gt_mel is not None, 'mels for stl targets not found'
        mel, f0, voiced, durations, dec_input = synthesis(model, data, alpha, guidance, style_token_weights, gt_mel if use_stl_target else None, choir_mode, choir_mode_variance, diffusion_model)
        
        # Save duration predictions
        df = utils.processed_to_df(data, join_syllables=False)
        df['dur_pred'] = durations[0]
        df.to_csv(os.path.join(result_dir, f'{filename}.csv'), index=False)

        # Save mel spec
        Image.fromarray(utils.spec_to_img(mel[0], f0[0], voiced[0])).save(os.path.join(result_dir, f'{filename}.png'))
        
        # Choir mode induces batches
        mel = mel.squeeze()
        f0 = f0.squeeze()
        voiced = voiced.squeeze()
        dec_input = dec_input.squeeze()
        
        f0 = audio.interpolate_f0(f0, voiced)

        if not save_norm_mel:
            mel = denorm_mel(mel)
        f0 = audio.denorm_note(f0)
        voiced = (1 / (1 + np.exp(-voiced))).round().astype(int)
        dec_input = np.array(dec_input)

        # Save mel numpy
        with open(os.path.join(result_dir, f'{filename}.pkl'), 'wb') as f:
            pickle.dump({
                'mel': mel,
                'f0': f0,
                'voiced': voiced,
                'dec_input': dec_input,
                'sample_prop': data.get('sample_prop', np.array([1.0]))},f)
        
        if use_gl:
            
            if choir_mode:
                gl_wav = mel_to_audio(mel[0])
                for i in range(1, choir_mode):
                    gl_wav += mel_to_audio(mel[i])
            else:
                gl_wav = mel_to_audio(mel)

            # Save griffin-lim audio
            sf.write(os.path.join(result_dir, f'{filename}.wav'), gl_wav, hp.sampling_rate, subtype='PCM_24')

def main(args):
    if not os.path.exists(args.result_path):
        os.mkdir(args.result_path)

    model = get_DNN(args.checkpoint_path)
    print(f'Loaded model with {utils.get_param_num(model)} parameters')

    diffusion_model = None
    if args.diffusion_decoder:
        diffusion_model = get_diff_model(args.diffusion_decoder)
        print(f'Loaded diffusion model with {utils.get_param_num(diffusion_model)} parameters')

    if not args.textnotegen:
        dataset = get_data(args.data_path)
    else:
        dataset = get_data_textnotegen(args.data_path, args.split_textnotegen)
    print(f'Loaded {len(dataset)} items from {args.data_path}')

    if args.style_token_target is not None:
        mel = load_mel(args.style_token_target)
        dataset = [(f,d,mel) for (f,d,_) in dataset]

    process(dataset, model, args.alpha, args.result_path, args.guidance, args.use_gl, args.style_token_weights, args.style_token_gt or args.style_token_target is not None, args.save_norm_mel, args.choir_mode, args.choir_mode_variance, diffusion_model)
    print('Processing done!')


if __name__ == "__main__":
    # Test
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=hp.label_path_test, help="Input folder")
    parser.add_argument('--textnotegen', type=bool, default=False, help="Whether the input data is in textnotegen format")
    parser.add_argument('--split_textnotegen', type=str, default='auto', help="Whether to split textnotegen inputs, can be either of none, all, auto")
    parser.add_argument('--guidance', type=str, default="none", help="Length regulator guidance, one of none, syllable, phoneme")
    parser.add_argument('--checkpoint_path', type=str, required=True, help="The model checkpoint to use")
    parser.add_argument("--alpha", type=float, default=1.0, help="Speed factor, >1 -> faster <1 slower")
    parser.add_argument('--result_path', type=str, default='results', help="Output folder")
    parser.add_argument('--use_gl', type=bool, default=False, help="Whether to use Griffin-Lim for mel inversion")
    parser.add_argument('--style_token_weights', type=float, default=None, help="Style token weights", nargs=hp.gst_token_num)
    parser.add_argument('--style_token_gt', type=bool, default=False, help="Whether to use gt mels to compute style token weights")
    parser.add_argument('--save_norm_mel', type=bool, default=False, help='Whether to save mels in the normalized internal representation')
    parser.add_argument('--choir_mode', type=int, default=None, help='Whether to use GST to synthesize several slightly different voices and then overlay them')
    parser.add_argument('--choir_mode_variance', type=float, default=0.5, help='How different the choir voices should be')
    parser.add_argument('--diffusion_decoder', type=str, default=None, help='An optional diffusion decoder checkpoint to use')
    args = parser.parse_args()

    print(args)
    assert args.guidance in ["none", "syllable", "phoneme"], "Invalid guidance parameter"
    assert args.split_textnotegen in ['none', 'all', 'auto'], "Invalid split_textnotegen parameter"
    assert not (args.textnotegen and args.guidance == 'phoneme'), "Textnotegen dataset does not provide phoneme information"
    assert not (args.textnotegen and args.style_token_gt), "Textnotegen dataset does not provide gt mels"
    assert os.path.exists(args.checkpoint_path), "Model checkpoint not found"
    assert args.alpha > 0 and args.alpha < 10, "Invalid alpha parameter"
    assert args.choir_mode is None or hp.use_gst, 'Can not use choir mode with deactivated GST'
    
    main(args)
    


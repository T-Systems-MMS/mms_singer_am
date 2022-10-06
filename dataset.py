import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import math
import time
import json
import pickle
import os
import warnings
from tqdm import tqdm

if __package__ is None or __package__ == '':
    import audio
    import hparams as hp
    from utils import process_text, pad_1D, pad_2D
    from utils import pad_1D_tensor, pad_2D_tensor
    import utils
else:
    from . import audio
    from . import hparams as hp
    from .utils import process_text, pad_1D, pad_2D, pad_1D_tensor, pad_2D_tensor
    from . import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

hop_length_ms = hp.hop_length / hp.sampling_rate
hop_length_frames = hp.sampling_rate / hp.hop_length


def load_labels(csv_name, mel_length=None):
    assert os.path.exists(csv_name)
    csv = pd.read_csv(csv_name)

    data = {
        'text': csv['PHONEME'].to_numpy().astype(int),
        'note': csv['PITCH'].to_numpy().astype(int),
        'duration': csv['DURATION_FRAMES'].to_numpy().astype(int),
        'syllable_duration_raw': csv['SYLLABLE_DURATION_FRAMES'].to_numpy().astype(int),
        'syllable_pos': csv['SYLLABLE_POS'].to_numpy().astype(int),
        'sample_prop': np.array([csv.at[0, 'SAMPLE_PROPABILITY']]) if 'SAMPLE_PROPABILITY' in csv.columns else np.array([1.0]),
        'phone_intensity': audio.norm_intensity(csv['INTENSITY'].to_numpy().astype(float)),
        'phone_hnr': audio.norm_hnr(csv['HNR'].to_numpy().astype(float)),
        'phone_f0': audio.norm_note(csv['F0'].to_numpy().astype(float)),
        'voiced_frac': csv['VOICED_FRAC'].to_numpy().astype(float)
    }

    if hp.bin_durations:
        data['syllable_duration'] = audio.bin_duration(data['syllable_duration_raw'])
    else:
        data['syllable_duration'] = data['syllable_duration_raw']

    # Either count src_pos from 1 to the last token (absolute positioning)
    # or relative to the syllable
    if not hp.relative_src_pos:
        data['src_pos'] = np.arange(len(data['text'])) + 1
    else:
        cur_pos = [1]
        counter = 1
        for last_p, cur_p in zip(data['syllable_pos'][:-1], data['syllable_pos'][1:]):
            counter += 1
            if last_p != cur_p:
                counter = 1
            cur_pos.append(counter)
        data['src_pos'] = np.array(cur_pos) 

    return data

def assert_data(key, data):
    assert len(data['text']) <= hp.max_seq_len_txt, f'{key} text sequence length too high'
    assert len(data['text']) == len(data['note'])
    assert len(data['text']) == len(data['syllable_duration'])
    assert len(data['text']) == len(data['syllable_pos'])
    assert len(data['text']) == len(data['duration'])
    assert len(data['text']) > 1, f'{key} text sequence length zero'

    assert np.all(data['text'] > 0), f'{key}: Text ID < 1 used'
    assert np.all(data['note'] > 0), f'{key}: Note ID < 1 used'
    assert np.all(data['syllable_duration'] > 0), f'{key}: Syllable duration ID < 1 used'
    assert np.all(data['duration'] > 0), f'{key}: Duration target < 1 used'
    assert np.all(data['syllable_pos'] > 0), f'{key}: Syllable pos < 1 used'
    assert np.all(data['syllable_pos'] < hp.syllable_pos_vocab_size), f'{key}: Syllable pos out of vocab size used'
    
    # assert np.all(data['energy'] > 0), f'{key}: Energy ID < 1 used'
    
    assert np.all(np.diff(data['syllable_pos']) <= 1), f'{key}: Syllable positions jump'
    assert np.all(np.diff(data['syllable_pos']) >= 0), f'{key}: Syllable positions are not ordered'
    assert np.all(data['text'] < hp.vocab_size), f'{key}: Text ID out of vocab'
    assert np.all(data['note'] < hp.note_vocab_size), f'{key}: Note ID out of vocab'
    assert np.all(data['src_pos'] < hp.phoneme_pos_vocab_size), f'{key}: Src pos out of vocab'
    assert np.all(data['mel_pos'] < hp.max_seq_len), f'{key}: Mel pos larger than seq len'
    
    if not hp.bin_durations:
        assert np.all(data['syllable_duration'] < hp.syllable_duration_vocab_size), f'{key}: Syllable duration out of vocab'
    elif np.any(data['syllable_duration_raw'] > hp.bin_durations_max):
        warnings.warn(f'{key}: Syllable duration {max(data["syllable_duration_raw"])} > binning range')
    if np.any(data['syllable_duration_raw'] < 0):
        warnings.warn(f'{key}: Syllable duration less than 0')
    
    
    assert abs(len(data['mel_target']) - sum(data['duration'])) < len(data['mel_target'])*0.02, f'{key}: Mel duration {len(data["mel_target"])} more than 2% off label durations {sum(data["duration"])}'
    if abs(len(data['mel_target']) - sum(data['duration'])) > 0:
        warnings.warn(f'{key}: Mel duration {len(data["mel_target"])} off label durations {sum(data["duration"])}')
    
    syldur = data['syllable_duration_raw'] * np.concatenate([[1], np.diff(data['syllable_pos'])]).astype(bool) # Only count each syllable duration once on a position change
    if sum(syldur) > hp.max_seq_len:
        warnings.warn(f'{key}: Cumulative syllable duration {sum(syldur)} greater than max mel sequence length {hp.max_seq_len}')
    if abs(sum(syldur) - len(data['mel_target'])) > len(data['mel_target'])*0.10:
        warnings.warn(f'{key}: Mel duration {len(data["mel_target"])} more than 10% off syllable durations {sum(syldur)}')
    
    syltemps = (~np.diff(data['syllable_pos']).astype(bool))*np.diff(data['syllable_duration'])
    assert np.all(syltemps == 0), f'{key}: Syllable durations are not consistent within single syllables'

    assert len(data['mel_target']) < hp.max_seq_len, f'{key}: Mel length {len(data["mel_target"])} larger than max seq len {hp.max_seq_len}!'

    assert sum(data['voiced_target']) > 0, f'{key}: No voiced frames'
    assert np.all(data['f0_target'][data['voiced_target'] != 0] != 0), f'{key}: F0 information missing for voiced parts'
    
    for k in data.keys():
        assert (not np.isnan(data[k]).any()), f'{key}: Nan in {k}'

    assert data['sample_prop'][0] >= 0 and data['sample_prop'][0] <= 1, f'{key}: Invalid sample propability: {data["sample_prop"]}'


def get_data_to_buffer(label_path=hp.label_path, mel_path=hp.mel_ground_truth, run_asserts=True):
    buffer = list()
    keys = [label.split('.')[0] for label in os.listdir(label_path)]
    
    keys.sort()

    start = time.perf_counter()
    for key in tqdm(keys):
        mel_gt_name = os.path.join(mel_path, f'csd-mel-{key}.pkl')

        with open(mel_gt_name, 'rb') as f:
            mel_data = pickle.load(f)

        csv_name = os.path.join(label_path, f'{key}.csv')
        data = load_labels(csv_name, mel_length=len(mel_data['mel']))

        data.update({
            'mel_target': mel_data['mel'],
            'mel_pos': np.arange(len(mel_data['mel'])) + 1,
            'f0_target': audio.norm_note(mel_data['f0']),
            'voiced_target': mel_data['voiced'],
            'frame_intensity': audio.norm_intensity(mel_data['intensity']),
            'frame_hnr': audio.norm_hnr(mel_data['hnr'])
        })
        
        if run_asserts:
            try:
                assert_data(key, data)
            except Exception as e:
                print(e)
                continue


        data = {k: torch.from_numpy(v) for k,v in data.items()}
        buffer.append(data)

    end = time.perf_counter()
    print("cost {:.2f}s to load all data into buffer.".format(end-start))

    return buffer


class BufferDataset(Dataset):
    def __init__(self, buffer):
        self.buffer = buffer
        self.weights = [b['sample_prop'].item() for b in buffer]
        self.length_dataset = len(self.buffer)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        if isinstance(idx, list):
            batch = [self.__getitem__(i) for i in idx]
            return collate_fn_pad(batch)
        return self.buffer[idx]

    def get_weights(self):
        return self.weights


def collate_fn_pad(batch):
    keys = ['text', 'mel_target', 'f0_target', 'voiced_target', 'duration', 'syllable_duration', 'syllable_duration_raw', 'syllable_pos', 'note', 'mel_pos', 'src_pos']
    output = {key:torch.nn.utils.rnn.pad_sequence([d[key] for d in batch], batch_first=True) for key in keys}
    output['mel_max_len'] = output['mel_target'].shape[1]
    return output

if __name__ == "__main__":
    # TEST
    test_ds = BufferDataset(get_data_to_buffer(run_asserts=True))
    test_dl = DataLoader(test_ds, batch_size = hp.batch_size, collate_fn = collate_fn_pad)
    print(f'Dataset size: {len(test_ds)}, dataloader size: {len(test_dl)}')
    max_mel_len = max([len(item["mel_target"]) for item in test_ds]) 
    print(f'Maximum mel length: {max_mel_len } (corresponds to {max_mel_len * hop_length_ms }s) ')
    min_mel_len = min([len(item["mel_target"]) for item in test_ds])
    print(f'Minimum mel length: {min_mel_len} (corresponds to {min_mel_len * hop_length_ms }s)')
    total_mel_len = sum([len(item["mel_target"]) for item in test_ds])
    print(f'Total mel length: {total_mel_len} (corresponds to {(total_mel_len * hop_length_ms)/3600 }h)')
    print(f'Maximum text length: {max([len(item["text"]) for item in test_ds])}')
    print(f'Minimum text length: {min([len(item["text"]) for item in test_ds])}')
    print(f'Maximum number of syllables: {max([sum(torch.diff(item["syllable_pos"]))+1 for item in test_ds])}')
    print(f'Highest note: {max([max(item["note"]) for item in test_ds])}')
    print(f'Lowest note: {min([min(item["note"][item["note"] != 1]) for item in test_ds])}')
    print(f'Highest phoneme id: {max([max(item["text"]) for item in test_ds])}')

    max_syllable_duration = max([max(item["syllable_duration_raw"]) for item in test_ds])
    print(f'Highest syllable duration (including unvoiced part at the beginning/end): {max_syllable_duration} (corresponds to {max_syllable_duration * hop_length_ms}s)')
    print(f'Avg syllable duration (including unvoiced part at the beginning/end): {np.mean([item["syllable_duration_raw"].float().mean() for item in test_ds])}')
    min_syllable_duration = min([min(item["syllable_duration_raw"]) for item in test_ds])
    print(f'Minimum syllable duration: {min_syllable_duration} (corresponts to {min_syllable_duration * hop_length_ms}s)')
    

    max_duration = max([max(item["duration"]) for item in test_ds])
    print(f'Highest duration: {max_duration} (corresponds to {max_duration * hop_length_ms}s)')
    avg_duration = np.mean([item['duration'].float().mean() for item in test_ds])
    print(f'Avg duration: {avg_duration}')
    min_duration = min([min(item["duration"]) for item in test_ds])
    print(f'Minimum duration: {min_duration} (corresponds to {min_duration * hop_length_ms}s)')

    weights = sum([item['sample_prop'][0] for item in test_ds])
    print(f'Sample weights sum to {weights}')
    f0_mean = np.mean([item['f0_target'].mean() for item in test_ds])
    print(f'Mean f0: {f0_mean}')

    print('Running dataset asserts')
    #get_data_to_buffer(run_asserts=True)
    #get_data_to_buffer(hp.label_path_test, hp.mel_ground_truth_test, run_asserts=True)
    print('No major assertions encountered, you are clear to start a training')

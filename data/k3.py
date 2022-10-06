
import numpy as np
import os
import json
import pickle
import pandas as pd
import warnings
import hparams as hp
import shutil
import math
from praatio import textgrid
import utils
import audio
from tqdm import tqdm
import multiprocessing
from data import data_utils

hop_length_ms = 1e-3 * hp.sampling_rate / hp.hop_length


def build_from_path(in_dir, target_dir):
    '''
    Preprocessing of CSD
    '''

    pool = multiprocessing.Pool(hp.preprocess_workers)
    
    print('Creating directories')
    
    snippets_dir = hp.dataset
    wav_snippets_dir = os.path.join(snippets_dir, 'wav_mono')
    csv_snippets_dir = os.path.join(snippets_dir, 'csv')
    mel_snippets_dir = os.path.join(snippets_dir, 'mel')
    
    os.makedirs(snippets_dir, exist_ok=True)
    os.makedirs(wav_snippets_dir, exist_ok=True)
    os.makedirs(csv_snippets_dir, exist_ok=True)    
    os.makedirs(mel_snippets_dir, exist_ok=True) 

    csv_dir = os.path.join(target_dir, 'csv')
    csv_in_dir = os.path.join(in_dir, 'csv')
    os.makedirs(csv_dir)
    wav_mono_dir = os.path.join(in_dir, 'wav_mono')

    print('Fixing K3 csvs')
    keys = [key.partition('.')[0] for key in os.listdir(csv_in_dir)]
    for key in tqdm(keys):
        _fix_k3_csv(csv_in_dir, csv_dir, key)

    if hp.max_snippet_length:
        print('Splitting snippets')
        keys = [key.partition('.')[0] for key in os.listdir(csv_dir)]
        keys = [(wav_mono_dir, csv_dir, snippets_dir, key) for key in keys]
        for _ in tqdm(pool.imap_unordered(data_utils.create_snippets_packed, keys), total=len(keys)):
            pass

    else:
        print('Splitting disabled')
        keys = [key.partition('.')[0] for key in os.listdir(csv_dir)]
        for key in tqdm(keys):
            shutil.copy(os.path.join(csv_dir, f'{key}.csv'),
                        os.path.join(csv_snippets_dir, f'{key}.csv'))
            shutil.copy(os.path.join(wav_mono_dir, f'{key}.wav'),
                        os.path.join(wav_snippets_dir, f'{key}.wav'))
    
    if hp.remove_nosounds:
        print(f'Removing {hp.remove_nosounds * 100}% of nosounds from labels')
        keys = [key.partition('.')[0] for key in os.listdir(csv_snippets_dir)]
        for key in tqdm(keys):
            data_utils.remove_nosounds(csv_snippets_dir, key, fraction=hp.remove_nosounds)
    
    if hp.max_note_length:
        print(f'Splitting too long notes into up to {hp.max_note_length}s snippets')
        keys = [key.partition('.')[0] for key in os.listdir(csv_snippets_dir)]
        for key in tqdm(keys):
            data_utils.split_long_notes(csv_snippets_dir, key, max_note_length=hp.max_note_length)


    # Test-train split
    if hp.test_train_split < 1:
        print('Test-train-split')
        keys = [key.partition('.')[0] for key in os.listdir(csv_snippets_dir)]
        
        # Deterministically split the keys into train and test indices
        keys.sort()
        np.random.seed(hp.test_train_split_seed)
        test_indices = np.random.choice(len(keys), int(len(keys) * (1-hp.test_train_split)), replace=False)
        
        snippets_dir_test = hp.dataset_test
        wav_snippets_dir_test = os.path.join(snippets_dir_test, 'wav_mono')
        csv_snippets_dir_test = os.path.join(snippets_dir_test, 'csv')
        mel_snippets_dir_test = os.path.join(snippets_dir_test, 'mel')

        os.makedirs(snippets_dir_test, exist_ok=True)
        os.makedirs(wav_snippets_dir_test, exist_ok=True)
        os.makedirs(csv_snippets_dir_test, exist_ok=True)
        os.makedirs(mel_snippets_dir_test, exist_ok=True)
        
        # Move over all test samples
        for idx in tqdm(test_indices):
            key = keys[idx]
            shutil.move(os.path.join(csv_snippets_dir, f'{key}.csv'),
                        os.path.join(csv_snippets_dir_test, f'{key}.csv'))
            shutil.move(os.path.join(wav_snippets_dir, f'{key}.wav'),
                        os.path.join(wav_snippets_dir_test, f'{key}.wav'))


    # Perform pitch shifting/time stretching if wanted
    if len(hp.augment_pitch_shift) or len(hp.augment_time_stretch):
        print('Performing data augmentation')
        keys = [key.partition('.')[0] for key in os.listdir(wav_snippets_dir)]
        keys = [(wav_snippets_dir, csv_snippets_dir, key, hp.augment_time_stretch, hp.augment_pitch_shift) for key in keys]
        for _ in tqdm(pool.imap_unordered(data_utils.perform_data_augmentation_packed, keys), total=len(keys)):
            pass

    # Process mels and F0
    print('Processing mels')
    keys = [key.partition('.')[0] for key in os.listdir(wav_snippets_dir)]
    keys = [(wav_snippets_dir, mel_snippets_dir, key) for key in keys]
    keys_test = [key.partition('.')[0] for key in os.listdir(wav_snippets_dir_test)]
    keys_test = [(wav_snippets_dir_test, mel_snippets_dir_test, key) for key in keys_test]
    keys = keys_test + keys
    for _ in tqdm(pool.imap_unordered(data_utils.process_utterance_packed, keys), total=len(keys)):
        pass

    print('Adding duration information')
    keys = [key.partition('.')[0] for key in os.listdir(wav_snippets_dir)]
    keys = [(csv_snippets_dir, mel_snippets_dir, key) for key in keys]
    keys_test = [key.partition('.')[0] for key in os.listdir(wav_snippets_dir_test)]
    keys_test = [(csv_snippets_dir_test, mel_snippets_dir_test, key) for key in keys_test]
    keys = keys_test + keys
    for args in tqdm(keys):
        data_utils.add_duration_labels(*args)

    print('Done')

def _fix_k3_csv(in_dir, out_dir, key):
    with open(hp.phoneme_dict, 'rb') as f:
        phoneme_dict = pickle.load(f)

    csv = pd.read_csv(os.path.join(in_dir, f'{key}.csv'))
    
    # Fix Nosound duration and position
    idx = csv['PHONEME'] == '<nosound>'
    csv.loc[idx, 'SYLLABLE_DURATION_FRAMES'] = csv.loc[idx, 'DURATION_FRAMES']

    last_pos = csv.loc[0, 'SYLLABLE_POS']
    for i in range(1, len(csv)):
        if csv.loc[i, 'PHONEME'] == '<nosound>':
            csv.loc[range(i, len(csv)), 'SYLLABLE_POS'] += 1
            new_pos = 0 if i == 0 else csv.loc[i-1, 'SYLLABLE_POS']
            csv.loc[i, 'SYLLABLE_POS'] = new_pos + 1

    csv['SYLLABLE_POS'] += 1 - csv.loc[0, 'SYLLABLE_POS']
    
    # Phoneme to phoneme_id
    csv['PHONEME_HUMAN'] = csv['PHONEME']
    csv['PHONEME'] = csv['PHONEME_HUMAN'].apply(lambda x: phoneme_dict[x])
    csv['PITCH'] = csv['PITCH'].astype(int)

    x = hp.hop_length / hp.sampling_rate * 1e3
    # Compute syllable start and end
    csv.loc[0, 'SYLLABLE_START'] = 0.0
    csv.loc[0, 'SYLLABLE_END'] = csv.loc[0, 'SYLLABLE_DURATION_FRAMES'] * x
    for i in range(1, len(csv)):
        if csv.loc[i-1, 'SYLLABLE_POS'] == csv.loc[i, 'SYLLABLE_POS']:
            csv.loc[i, 'SYLLABLE_START'] = csv.loc[i-1, 'SYLLABLE_START']
            csv.loc[i, 'SYLLABLE_END'] = csv.loc[i-1, 'SYLLABLE_END']
        else:
            csv.loc[i, 'SYLLABLE_START'] = csv.loc[i-1, 'SYLLABLE_END']
            csv.loc[i, 'SYLLABLE_END'] = csv.loc[i, 'SYLLABLE_START'] + csv.loc[i, 'SYLLABLE_DURATION_FRAMES'] * x 
    
    # Pitch min = 1, rename columns
    csv.loc[csv['PITCH'] == 0, 'PITCH'] = 1
    csv.rename({'START': 'MFA_ADJ_START', 'END': 'MFA_ADJ_END'}, axis=1, inplace=True)

    # Fix double nosounds
    while not np.all([csv['VOICED'][i+1] or csv['VOICED'][i] for i in range(len(csv)-1)]):
        for i in range(1, len(csv)):
            if not csv.loc[i, 'VOICED'] and not csv.loc[i-1, 'VOICED']:
                csv.loc[i-1, 'MFA_ADJ_END'] = csv.loc[i, 'MFA_ADJ_END']
                csv.loc[i-1, 'SYLLABLE_END'] = csv.loc[i, 'SYLLABLE_END']
                csv.loc[i-1, 'DURATION_MS'] += csv.loc[i, 'DURATION_MS']
                csv.loc[i-1, 'DURATION_FRAMES'] += csv.loc[i, 'DURATION_FRAMES']
                csv.loc[i-1, 'SYLLABLE_DURATION_FRAMES'] += csv.loc[i, 'SYLLABLE_DURATION_FRAMES']
                csv.drop(i, inplace=True)
                csv.reset_index(inplace=True, drop=True)
                break
    pos_arr = csv['SYLLABLE_POS'].unique()
    pos_arr.sort()
    csv['SYLLABLE_POS'] = csv['SYLLABLE_POS'].apply(lambda x: np.where(pos_arr==x)[0][0] + 1)
    
    # Rescale times
    csv['MFA_ADJ_START'] /= 1000
    csv['MFA_ADJ_END'] /= 1000
    csv['SYLLABLE_START'] /= 1000
    csv['SYLLABLE_END'] /= 1000
    
    csv.to_csv(os.path.join(out_dir, f'{key}.csv'), index=False)

def _copy_files(file_path, target_dir):
    shutil.copy2(os.path.join(file_path), target_dir)
    return


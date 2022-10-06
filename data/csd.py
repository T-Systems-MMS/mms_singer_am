from __future__ import annotations
from cProfile import label
import numpy as np
import os
import json
import pickle
import librosa
import pandas as pd
import warnings
import scipy
import hparams as hp
import shutil
import math
import mido
from praatio import textgrid
from collections import Counter
import matplotlib.pyplot as plt
import utils
import audio
from tqdm import tqdm
#import parselmouth
import multiprocessing
import soundfile as sf
from data import data_utils

hop_length_ms = 1e-3 * hp.sampling_rate / hp.hop_length


def build_from_path(in_dir, target_dir):
    '''
    Preprocessing of CSD
    '''

    pool = multiprocessing.Pool(hp.preprocess_workers)
    assert os.path.exists(os.path.join(target_dir, 'mfa_output')), 'MFA output missing, please run data/csd_preprocess.sh'
    assert os.path.exists(os.path.join(target_dir, 'mfa_csd_corpus')), 'MFA corpus missing, please run data/csd_preprocess.sh'

    print('Creating directories')
    wav_mono_dir = os.path.join(target_dir, 'wav_mono')
    csv_dir = os.path.join(target_dir, 'csv')
    textgrid_dir = os.path.join(target_dir, 'textgrid')

    os.makedirs(wav_mono_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(textgrid_dir, exist_ok=True)
    
    snippets_dir = hp.dataset
    wav_snippets_dir = os.path.join(snippets_dir, 'wav_mono')
    csv_snippets_dir = os.path.join(snippets_dir, 'csv')
    mel_snippets_dir = os.path.join(snippets_dir, 'mel')
    
    os.makedirs(snippets_dir, exist_ok=True)
    os.makedirs(wav_snippets_dir, exist_ok=True)
    os.makedirs(csv_snippets_dir, exist_ok=True)    
    os.makedirs(mel_snippets_dir, exist_ok=True) 

    #phoneme_dict = _create_phoneme_dict(os.path.join(target_dir, 'mfa_output'), target_dir)
    if not os.path.exists(os.path.join('data', 'csd_phoneme_dict.pkl')):
        csd_phoneme_dict = _create_csd_phoneme_dict(os.path.join(in_dir, 'csv'), target_dir)
    else:
        with open(os.path.join('data', 'csd_phoneme_dict.pkl'), 'rb') as f:
            csd_phoneme_dict = pickle.load(f)
    mfa_csd_dict = _read_mfa_csd_dict(os.path.join('data', 'mfa_csd_dict.txt'))

    
    # enhance existing labels (csv format)
    # the copy functions shall later be replaced by the actual functions from make_mfa_corpus.py and Montreal Forced Aligner
    # create short snippets (csv+wav) from the original CSD
    print('Processing labels')
    keys = [key.partition('.')[0] for key in os.listdir(os.path.join(in_dir, 'csv'))]
    for key in keys:
        _copy_files(os.path.join(target_dir, 'mfa_csd_corpus', f'{key}.wav'), wav_mono_dir)
        _copy_files(os.path.join(target_dir, 'mfa_output', f'{key}.TextGrid'), textgrid_dir)
    
    keys = [(textgrid_dir, os.path.join(in_dir, 'csv'), csv_dir, key, csd_phoneme_dict, mfa_csd_dict) for key in keys]
    for _ in tqdm(pool.imap_unordered(_process_label_inv_packed, keys), total=len(keys)):
        pass
        #_process_label_inv(textgrid_dir, os.path.join(in_dir, 'csv'), csv_dir, key, csd_phoneme_dict, mfa_csd_dict)
        #_process_label(textgrid_dir, os.path.join(in_dir, 'mid'), csv_dir, key, phoneme_dict)
        

    if hp.max_snippet_length:
        print('Splitting snippets')
        keys = [key.partition('.')[0] for key in os.listdir(os.path.join(in_dir, 'csv'))]
        keys = [(wav_mono_dir, csv_dir, snippets_dir, key) for key in keys]
        for _ in tqdm(pool.imap_unordered(data_utils.create_snippets_packed, keys), total=len(keys)):
            pass
    else:
        print('Splitting disabled')
        keys = [key.partition('.')[0] for key in os.listdir(os.path.join(in_dir, 'csv'))]
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

def _create_csd_phoneme_dict(csv_dir, target_dir):
    phonemes = []
    for key in os.listdir(csv_dir):
        df = pd.read_csv(os.path.join(csv_dir, key))
        for syll in df['syllable']:
            for ph in syll.split('_'):
                phonemes.append(ph)
    phonemes = list(set(phonemes))
    phonemes.insert(0, '<pad>')
    phonemes.insert(1, '<nosound>')

    phoneme_dict = {phonemes[i]: i for i in range(0, len(phonemes))}

    # save the dictionary
    with open(os.path.join(target_dir, 'csd_phoneme_dict.pkl'), 'wb') as f:
        pickle.dump(phoneme_dict, f)

    return phoneme_dict

def _read_mfa_csd_dict(file):
    with open(file, 'r') as f:
        data = f.readlines()
    d = {l.split(' ')[0].strip():l.split(' ')[1].strip() for l in data}
    d['<nosound>'] = '<nosound>'
    return d
    

def _copy_files(file_path, target_dir):
    shutil.copy2(os.path.join(file_path), target_dir)
    return

def _process_label_inv_packed(args):
    _process_label_inv(*args)

def _process_label_inv(in_dir, csv_dir, target_dir, key, phoneme_dict, mfa_csd_dict):
    csv = pd.read_csv(os.path.join(csv_dir, f'{key}.csv'))
    def make_nosound_df(start, end):
        return pd.DataFrame({
            'start': start,
            'end': end,
            'pitch': 1,
            'syllable': '<nosound>'}, index=[0])

    # insert <nosound> at beginning
    if csv['start'][0] > 0:
        csv = pd.concat([make_nosound_df(0, csv['start'][0]), csv]).reset_index(drop=True)
    if csv['start'][0] < 0:
        csv.loc[0, 'start'] = 0

    # insert <nosound> at all breaks
    idx = 1
    while idx < len(csv):
        if csv['end'][idx-1] != csv['start'][idx]:
            csv = pd.concat([
                    csv.iloc[0:idx],
                    make_nosound_df(csv['end'][idx-1], csv['start'][idx]),
                    csv.iloc[idx:]
                ]).reset_index(drop=True)
        idx += 1

    assert np.all([csv.at[i, 'end'] == csv.at[i+1, 'start'] for i in range(len(csv)-1)])

    # Load MFA annotations
    mfa = textgrid.openTextgrid(os.path.join(in_dir, f'{key}.TextGrid'), True)
    mfa = pd.DataFrame(mfa.tierDict['phones'].entryList, columns=['start', 'end', 'phoneme'])
    mfa['phoneme'] = mfa['phoneme'].apply(lambda x: x if x else '<nosound>')

    # Insert <nosound> at the end of csv
    if csv['end'].iloc[-1] < mfa['end'].iloc[-1]:
        csv = pd.concat([csv,
            make_nosound_df(csv['end'].iloc[-1], mfa['end'].iloc[-1]),
        ]).reset_index(drop=True)

    # Explode dataframe to phoneme level
    csv['syllable_pos'] = [i+1 for i in range(len(csv))]
    csv['phoneme'] = csv['syllable'].apply(lambda x: x.split('_'))
    csv = csv.explode('phoneme').reset_index(drop=True)
    
    # Run through the CSV and match mfa entries
    # Mark matched entries in the mfa
    mfa['marked'] = [False for _ in range(len(mfa))]
    csv['mfa_start'] = [np.nan for _ in range(len(csv))]
    csv['mfa_end'] = [np.nan for _ in range(len(csv))]
    
    def match(csv_row, mfa_row):
        return max(csv_row['start'], mfa_row['start']) <= min(csv_row['end'], mfa_row['end']) and \
            mfa_csd_dict[csv_row['phoneme']] == mfa_row['phoneme'] and \
            not mfa_row['marked']

    for idx, row in csv.iterrows():
        mfa_match = mfa[mfa.apply(lambda x: match(row, x), axis=1)]
        if len(mfa_match):
            mfa.at[mfa_match.index[0], 'marked'] = True
            csv.at[idx, 'mfa_start'] = mfa_match.iloc[0]['start']
            csv.at[idx, 'mfa_end'] = mfa_match.iloc[0]['end']
    
    # In rare occasions, the MFA could not match a syllable and some rows in csv will be NaN
    # In these cases, drop the csv items
    if csv['mfa_start'].isnull().values.any():
        # Fix durations
        indices = np.arange(len(csv))[csv['mfa_start'].isnull()]
        for idx in indices:
            if idx > 0:
                csv.loc[csv['syllable_pos'] == csv.loc[idx-1, 'syllable_pos'], 'end'] = csv.loc[idx, 'end']
        #warnings.warn(f'{key}: MFA could not align all syllables')
        # Drop unaligned syllables
        csv = csv[~csv['mfa_start'].isnull()].reset_index(drop=True)

        # Fix syllable positions
        syl_pos = np.sort(csv['syllable_pos'].unique())
        syl_pos = dict(zip(syl_pos, np.arange(len(syl_pos))+1))
        csv['syllable_pos'] = csv['syllable_pos'].apply(lambda x: syl_pos[x])

    # Run through the mfa list and add the times of unmarked phonemes to the neighbors
    # The strategy dictates how much of the unmatched duration to add to the previous one and how much to the next one
    # (0.5, 0.5) means split in half, (1, 0) means all to the previous
    match_strategy = (0.5, 0.5)
    # First deal with first and last item
    while mfa.at[0, 'marked'] == False:
        assert len(mfa) > 1
        mfa.at[1, 'start'] = mfa.at[0, 'start']
        mfa = mfa[1:].reset_index(drop=True)
    while mfa.iloc[-1]['marked'] == False:
        assert len(mfa) > 1
        idx = len(mfa)-1
        mfa.at[idx-1, 'end'] = mfa.at[idx, 'end']
        mfa = mfa[:-1].reset_index(drop=True)
    

    # Spread unmarked items to the neighbouring items
    durations = mfa.apply(lambda x: x['end']-x['start'] if not x['marked'] else 0, axis=1)
    durations = pd.concat([pd.Series([0]), durations, pd.Series([0])]).reset_index(drop=True)
    mfa['end'] += (durations.iloc[2:]*match_strategy[0]).reset_index(drop=True)
    mfa['start'] -= (durations.iloc[:-2]*match_strategy[1]).reset_index(drop=True)
    mfa = mfa[mfa['marked']].reset_index(drop=True)


    # Run through the matching again, all items will be matched
    # Hence, mfa_adj times will be gapless
    mfa['marked'] = [False for _ in range(len(mfa))]
    csv['mfa_adj_start'] = [np.nan for _ in range(len(csv))]
    csv['mfa_adj_end'] = [np.nan for _ in range(len(csv))]
    for idx, row in csv.iterrows():
        mfa_match = mfa[mfa.apply(lambda x: match(row, x), axis=1)]
        if len(mfa_match):
            mfa.at[mfa_match.index[0], 'marked'] = True
            csv.at[idx, 'mfa_adj_start'] = mfa_match.iloc[0]['start']
            csv.at[idx, 'mfa_adj_end'] = mfa_match.iloc[0]['end']



    
    # Prepare split list, split in the middle of <nosound>
    #timestamps = (csv['mfa_adj_start'] + csv['mfa_adj_end']) / 2
    #timestamps = pd.concat([timestamps, pd.Series([csv['mfa_adj_end'].iloc[-1]])])
    #split_indices = np.arange(len(csv))[csv['phonemes']=='<nosound>']
    #split_indices = utils.find_optimal_split(timestamps, split_indices, hp.max_snippet_length)
    #split_timestamps = timestamps[split_indices]

    # Make it compatible with previous csv style
    csv['energy'] = [0 for _ in range(len(csv))]
    csv['voiced'] = csv['phoneme'] != '<nosound>'
    csv['phoneme'] = csv['phoneme'].apply(lambda x: phoneme_dict[x])
    csv['syllable_start'] = csv['start']
    csv['syllable_end'] = csv['end']
    csv.drop('start', axis=1, inplace=True)
    csv.drop('end', axis=1, inplace=True)

    # Who started with this uppercase stuff?
    csv.columns= csv.columns.str.upper()

    csv.to_csv(os.path.join(target_dir, f'{key}.csv'), index=False)

    return csv, mfa

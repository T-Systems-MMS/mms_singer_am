import pandas as pd
import numpy as np
import hparams as hp
import librosa
import pickle
import shutil
import os
import parselmouth
import multiprocessing
from praatio import textgrid
from tqdm import tqdm
from data import  data_utils

hop_length_ms = 1e-3 * hp.sampling_rate / hp.hop_length


def get_length_of_wav(wav_path: str):
    return librosa.get_duration(filename=wav_path) * 1e3


def note_conversion(frequency: float):
    note_conversion = pd.read_csv(os.path.join('data', 'note_conversion.csv'), header=None)
    note_conversion = dict(zip(note_conversion[3], note_conversion[0]))

    frequency = min(list(note_conversion.keys()), key=lambda x: abs(x - frequency))
    return note_conversion[frequency]


def preprocess_speech(textgrid_path: str, wav_path: str):
    mfa = textgrid.openTextgrid(textgrid_path, True)
    mfa = pd.DataFrame(mfa.getTier('phones').entries, columns=['START', 'END', 'PHONEME'])
    f0 = data_utils.read_f0_parselmouth(wav_path)
    
    with open(hp.phoneme_dict, 'rb') as f:
        phoneme_dict = pickle.load(f)
    
    mfa['PHONEME'] = mfa['PHONEME'].apply(lambda x: x if x else '<nosound>')
    mfa['SYLLABLE'] = mfa['PHONEME']
    mfa['PHONEME'] = mfa['PHONEME'].map(phoneme_dict)
    mfa['MFA_ADJ_START'] = mfa['START']
    mfa['MFA_ADJ_END'] = mfa['END']
    mfa['SYLLABLE_START'] = mfa['START']
    mfa['SYLLABLE_END'] = mfa['END']
    mfa['DURATION_MS'] = mfa['END']*1e3 - mfa['START']*1e3
    mfa['DURATION_FRAMES'] = (mfa['END'] * hop_length_ms * 1e3).round().astype(int)
    durations = mfa['DURATION_FRAMES'].to_numpy()
    durations[1:] -= durations[:-1].copy()
    mfa['DURATION_FRAMES'] = durations
    mfa['SYLLABLE_POS'] = mfa.index + 1
    mfa['SYLLABLE_DURATION_FRAMES'] = mfa['DURATION_FRAMES']
    mfa['VOICED'] = mfa['SYLLABLE'] != '<nosound>'
    mfa['ENERGY'] = 0
    mfa['PITCH'] = mfa.apply(lambda x: get_f0_at_interval(f0, x['START'], x['END']), axis=1)
    mfa['PITCH'] = mfa['PITCH'].apply(lambda freq: note_conversion(freq))

    end_time_mfa = mfa['END'][mfa.index[-1]] * 1e3
    if end_time_mfa != get_length_of_wav(wav_path):
        print(f'annotations do not match audio length for {wav_path}')
        return pd.DataFrame()
    
    if end_time_mfa != mfa['DURATION_MS'].sum():
        print(f'sum of durations of annotations do not match audio length for {wav_path}')
        return pd.DataFrame()

    return mfa


def get_f0_at_interval(f0, start, end):
    start = int(round(start * hop_length_ms))
    end = int(round(end * hop_length_ms))
    f0 = f0[start:end, 2]
    
    if len(f0) == 0:
        return 0
    
    return sum(f0) / len(f0)

def preprocess_single_item(args):
    input_dir, output_dir, key = args

    textgrid_path = os.path.join(input_dir, 'TextGrid', f'{key}.TextGrid') 
    wav_path = os.path.join(input_dir, 'wavs', f'{key}.wav') 
    result = preprocess_speech(textgrid_path, wav_path)

    if len(result.index) == 0:
        return key
    else:
        result.to_csv(os.path.join(output_dir, 'csv', f'{key}.csv'), index=False)
        shutil.copy(os.path.join(input_dir, 'wavs', f'{key}.wav'),
                    os.path.join(output_dir, 'wav_mono', f'{key}.wav'))
        data_utils.process_utterance(os.path.join(output_dir, 'wav_mono'),
                                     os.path.join(output_dir, 'mel'),
                                     key)
        data_utils.add_duration_labels(os.path.join(output_dir, 'csv'),
                                       os.path.join(output_dir, 'mel'),
                                       key)
        return None

def build_from_path(input_dir, output_dir):

    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    output_dir = os.path.join(output_dir, 'snippets')
    os.makedirs(output_dir)

    csv_snippets_dir = os.path.join(output_dir, 'csv')
    wav_snippets_dir = os.path.join(output_dir, 'wav_mono')
    mel_snippets_dir = os.path.join(output_dir, 'mel')
    os.makedirs(csv_snippets_dir, exist_ok=True)
    os.makedirs(wav_snippets_dir, exist_ok=True)
    os.makedirs(mel_snippets_dir, exist_ok=True)
    
    ignore_list = []

    keys = [key.partition('.')[0] for key in os.listdir(os.path.join(input_dir, 'wavs'))]
    keys.sort()
    keys = [(input_dir, output_dir, key) for key in keys]

    pool = multiprocessing.Pool(32)
    for res in tqdm(pool.imap_unordered(preprocess_single_item, keys), total=len(keys), smoothing=0.01):
        if res is not None:
            ignore_list.append(res)

    print('where preprocessing failed:', ignore_list)
    


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
            shutil.move(os.path.join(mel_snippets_dir, f'csd-mel-{key}.pkl'),
                        os.path.join(mel_snippets_dir_test, f'csd-mel-{key}.pkl'))

    print('Done')


def main():
    input_dir = os.path.join('data', 'LJSpeech-1.1')
    output_dir = os.path.join('data', 'LJSpeech_processed')
    preprocess_LJSpeech(input_dir, output_dir)


if __name__ == '__main__':
    main()

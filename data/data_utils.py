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
from pydub import AudioSegment
#import parselmouth # Can not use this because of license
import multiprocessing
import soundfile as sf

hop_length_ms = 1e-3 * hp.sampling_rate / hp.hop_length

def read_f0_parselmouth(wav_path):
    """Read F0 information from a wav file using praat parselmouth

    Returns an array with 5 x n_frames size, where 5 corresponds to the dimensions
    time_step, voiced, F0, intensity and harmonics-to-noise-ratio
    """
    snd = parselmouth.Sound(wav_path)
    time_step = hp.hop_length / hp.sampling_rate
    pitches = snd.to_pitch(time_step = time_step)
    intensities = snd.to_intensity(time_step = time_step).values[0]
    if len(pitches) > len(intensities):
        intensities = np.pad(intensities, (0, len(pitches) - len(intensities)))
    elif len(pitches) < len(intensities):
        intensities = intensities[0:len(pitches)]
    
    hnrs = snd.to_harmonicity(time_step = time_step).values[0]
    if len(pitches) > len(hnrs):
        hnrs = np.pad(hnrs, (0, len(pitches) - len(hnrs)))
    elif len(pitches) < len(hnrs):
        hnrs = hnrs[0:len(pitches)]
    hnrs[hnrs<0] = 0

    f0 = np.stack([
        [i*time_step for i in range(len(pitches))], # Time steps
        pitches.selected_array['frequency'] != 0,   # Voiced
        pitches.selected_array['frequency'],        # F0
        intensities,                                # Intensity
        hnrs                                        # Harmonics to noise ratio
    ]).transpose()

    # Interpolate missing f0 values at unvoiced sections to stabilize pitch cwt
    last_value = f0[np.argmax(f0[:,2] > 0), 2]
    for i in range(1, len(f0)):
        if f0[i, 2] == 0: #not voiced
            f0[i, 2] = last_value #f0
        else:
            last_value = f0[i,2]

    return f0

def read_f0_librosa(wave_audio):
    """Read F0 information using librosa, returns the same format as read_f0_parselmouth
    This is slower, but yields better F0 estimates. HNR estimation is not implemented yet
    """
    f0, voiced, voiced_prop = librosa.pyin(y=wave_audio,
                    fmin=hp.mel_fmin,
                    fmax=hp.mel_fmax/3,
                    sr=hp.sampling_rate,
                    frame_length=hp.win_length*2,
                    win_length=hp.win_length,
                    hop_length=hp.hop_length,
                    center=True)

    f0[np.isnan(f0)] = 0.0
    intensity = librosa.feature.rms(y=wave_audio, frame_length=hp.win_length, hop_length=hp.hop_length, center=True)[0,:]
    time_step = hp.hop_length / hp.sampling_rate
    f0 = np.stack([
        [i*time_step for i in range(len(f0))], # Time steps
        voiced.astype(float),                  # Voiced
        f0,                                    # F0
        intensity,                             # Intensity
        np.zeros(len(f0))                      # Harmonics to noise ratio (unavailable in librosa)
    ]).transpose()
    
    # Interpolate missing f0 values at unvoiced sections to stabilize pitch cwt
    last_value = f0[np.argmax(f0[:,2] > 0), 2]
    for i in range(1, len(f0)):
        if f0[i, 2] == 0: #not voiced
            f0[i, 2] = last_value #f0
        else:
            last_value = f0[i,2]

    return f0

# Wrapper for pool.imap_unordered
def process_utterance_packed(args):
    process_utterance(*args)

def process_utterance(wav_path, out_dir, key):
    """Computes a mel-scale spectrogram and f0 information from a wave file
    
    Writes a pickled dictionary to {out_dir}/csd-mel-{key}.pkl 
    """
    # Compute a mel-scale spectrogram from the wav:
    wav_path = os.path.join(wav_path, f'{key}.wav')
    wave_audio,_ = librosa.load(wav_path, sr=hp.sampling_rate, mono=True)

    #mel_spectrogram = audio.tools.get_mel(wav_path).numpy().astype(np.float32).transpose()
    mel_spectrogram = audio.norm_mel(audio.audio_to_mel(wave_audio))

    # read f0 information from reaper output
    # After processing, the three columns of the f0 array are
    # (timestamp, V/UV, F0)
    mel_timestamps = [(hp.hop_length/hp.sampling_rate) * i for i in range(len(mel_spectrogram))]
    #f0 = read_f0_parselmouth(wav_path)
    f0 = read_f0_librosa(wave_audio)

    assert abs(mel_timestamps[-1] - f0[-1,0]) < 0.1, 'Mel length and f0 length mismatch'
   
    assert not np.isnan(f0).any(), f'NaN encountered: {f0}'
    # Interpolate f0 information to the mel timescale
    interpol = scipy.interpolate.interp1d(f0[:,0], f0, 'nearest', axis=0, bounds_error=False, fill_value=np.array([f0[-1,0], 0, 0, f0[-1, 3], f0[-1, 4]]))
    f0 = interpol(mel_timestamps)

    data = {
        'mel': mel_spectrogram,
        'f0': f0[:,2],
        'voiced': f0[:,1].astype(bool),
        'intensity': f0[:, 3],
        'hnr': f0[:,4],
    }

    # Write the spectrograms to disk:
    mel_filename = f'csd-mel-{key}.pkl'
    with open(os.path.join(out_dir, mel_filename), 'wb') as f:
        pickle.dump(data, f)

    return data

def perform_data_augmentation_packed(args):
    perform_data_augmentation(*args)

# Augment a single file by a given warp factor
def perform_data_augmentation(wav_snippets_dir, csv_snippets_dir, key, time_stretch, pitch_shift):
    """Performs time-stretching and pitch-shifting according to the settings in hparams.py

    Writes multiple new .wav and .csv files to the wav and csv folders
    """
    wav, sr = librosa.load(os.path.join(wav_snippets_dir, f'{key}.wav'), sr=hp.sampling_rate)
    assert wav.shape[-1] > 0, f'{key} wave file has length 0'
    labels = pd.read_csv(os.path.join(csv_snippets_dir, f'{key}.csv'))
    
    for warp, prop in time_stretch:
        tmp = librosa.effects.time_stretch(wav, rate=warp)
        warp_str = str(warp).replace('.', '-')
        output_file = key + '_timestretched' + warp_str + '.wav'
        sf.write(os.path.join(wav_snippets_dir, output_file), tmp, sr, 'PCM_24')
        
        old_duration = labels.iloc[-1].MFA_ADJ_END
        new_duration = librosa.get_duration(filename=os.path.join(wav_snippets_dir, output_file))
        effective_warp = old_duration / new_duration # copes for librosa/soundfile being unprecise
        
        labels_copy = labels.copy()
        labels_copy['SAMPLE_PROPABILITY'] = [prop for _ in range(len(labels_copy))]
        labels_copy['SYLLABLE_START'] /= effective_warp
        labels_copy['SYLLABLE_END'] /= effective_warp
        labels_copy['MFA_ADJ_START'] /= effective_warp
        labels_copy['MFA_ADJ_END'] /= effective_warp
        labels_copy.to_csv(os.path.join(csv_snippets_dir, f'{key}_timestretched{warp_str}.csv'), index=None)

    for pitch, prop in pitch_shift:
        tmp = librosa.effects.pitch_shift(wav, sr=sr, n_steps=pitch)
        sf.write(os.path.join(wav_snippets_dir, f'{key}_pitchshifted{pitch}.wav'), tmp, sr, 'PCM_24')
        
        labels_copy = labels.copy()
        labels_copy['SAMPLE_PROPABILITY'] = [prop for _ in range(len(labels_copy))]
        labels_copy['PITCH'] = labels_copy['PITCH'].apply(lambda x: x if x == 1 else x+pitch)
        labels_copy.to_csv(os.path.join(csv_snippets_dir, f'{key}_pitchshifted{pitch}.csv'), index=None)


def remove_nosounds(csv_dir, key, fraction=1):
    """
    Removes a certain fraction of <nosound> labels and appends the time to the previous note.
    This way, the number of <nosound>s might represent inference time conditions more
    """
    start_key = 'MFA_ADJ_START'
    end_key = 'MFA_ADJ_END'

    csv = pd.read_csv(os.path.join(csv_dir, f'{key}.csv'))
    
    # Find the shortest nosounds and their index in the original array
    nosound_indices = np.arange(len(csv))[csv['PHONEME'] == 1]
    durations = csv[end_key] - csv[start_key]
    durations = durations.iloc[nosound_indices].reset_index(drop=True)
    tmp_df = pd.DataFrame()
    tmp_df['idx'] = nosound_indices
    tmp_df['dur'] = durations
    tmp_df = tmp_df[tmp_df['idx'] > 0]

    # Discart the longest nosounds
    tmp_df = tmp_df.sort_values(by='dur', ignore_index=True, ascending=True)
    tmp_df = tmp_df.iloc[0:int(len(tmp_df)*fraction)]
    tmp_df = tmp_df.sort_values(by='idx', ignore_index=True, ascending=False)
    
    # Update the original dataframe
    # Append the nosounds to the last phoneme in the previous syllable
    for idx in tmp_df['idx']:
        csv.loc[idx-1, 'MFA_ADJ_END'] = csv.loc[idx, 'MFA_ADJ_END']
        csv.loc[csv['SYLLABLE_POS'] == csv.loc[idx-1, 'SYLLABLE_POS'], 'SYLLABLE_END'] = csv.loc[idx, 'SYLLABLE_END']

    # Update the original array
    sound_indices = np.arange(len(csv))[~np.isin(np.arange(len(csv)), tmp_df['idx'])]
    csv = csv.iloc[sound_indices]
    
    # Update syllable positions
    syl_pos = np.sort(csv['SYLLABLE_POS'].unique())
    syl_pos = dict(zip(syl_pos, np.arange(len(syl_pos))+1))
    csv['SYLLABLE_POS'] = csv['SYLLABLE_POS'].apply(lambda x: syl_pos[x])

    csv.to_csv(os.path.join(csv_dir, f'{key}.csv'), index=False)

    return csv

def split_long_notes(csv_dir, key, max_note_length):
    """If there are notes longer than a threshold in the data, split them into multiple shorter ones.
    In our case, we knew that notes during inference would never exceed this length so never showing the model notes this long makes the
    duration predictor generalize better.
    """
    csv = pd.read_csv(os.path.join(csv_dir, f'{key}.csv'))

    # Splitting only works for notes that have a single phoneme
    # Add a temporary column
    csv['sylcount'] = csv['SYLLABLE_POS'].apply(lambda x: sum(csv['SYLLABLE_POS'] == x))

    # Split notes that are too long into multiple smaller ones
    long_notes = csv[((csv['SYLLABLE_END'] - csv['SYLLABLE_START']) > max_note_length) & (csv['sylcount'] == 1)]


    offset = 0
    for idx, row in long_notes.iterrows():
        splitcount = math.ceil((row['SYLLABLE_END'] - row['SYLLABLE_START'])/max_note_length)
        splitdur = (row['SYLLABLE_END'] - row['SYLLABLE_START'])/splitcount
        splitdur_mfa_adj = (row['MFA_ADJ_END'] - row['MFA_ADJ_START'])/splitcount
        row_new = pd.DataFrame([dict(row, 
            SYLLABLE_START=row['SYLLABLE_START']+splitdur*i, 
            SYLLABLE_END=row['SYLLABLE_START']+splitdur*(i+1),
            MFA_ADJ_START=row['MFA_ADJ_START']+splitdur_mfa_adj*i,
            MFA_ADJ_END=row['MFA_ADJ_START']+splitdur_mfa_adj*(i+1),
            SYLLABLE_POS=row['SYLLABLE_POS']+i+offset) for i in range(splitcount)])
        # Insert the new row into the csv dataframe while dropping the previous element at that position
        idx += offset
        csv.loc[idx+1:, 'SYLLABLE_POS'] += splitcount-1
        csv = pd.concat([csv[0:idx], row_new, csv[idx+1:]]).reset_index(drop=True)
        offset += splitcount - 1

    csv = csv.drop('sylcount', axis=1)
    csv.to_csv(os.path.join(csv_dir, f'{key}.csv'), index=False)
    return csv


def create_snippets_packed(args):
    create_snippets(*args)

def create_snippets(wav_dir, csv_dir, target_dir, key):
    '''
    Create snippets (csv annotations + wav mono) of given dataset
    Uses a tree search to find an optimal split for each file, where preferrably only unvoiced sections are used as splitting locations
    '''
    annotations = pd.read_csv(os.path.join(csv_dir, f'{key}.csv'))
    split_times = utils.find_optimal_split_2(list(zip(list(annotations['MFA_ADJ_START']), list(annotations['MFA_ADJ_END']))), 
                         list(zip(list(annotations['SYLLABLE_START']), list(annotations['SYLLABLE_END']))),
                         list(annotations['VOICED']),
                         hp.max_snippet_length, hp.max_note_length, hp.max_snippet_length*0.33)
    
    #split_times = find_split_times(annotations, max_length=hp.max_snippet_length)
    # TODO fix this bug
    #if np.any([start < 0 for (start, _) in split_times]):
    #    warnings.warn(f'{key}: Split start found at negative timestamp, this is a bug: {split_times}')
    #    split_times = [(start if start>0 else 0, end) for (start, end) in split_times]

    split_annotations(annotations=annotations, key=key, target_dir=target_dir, split_times=split_times)
    split_wav(in_dir=wav_dir, key=key, target_dir=target_dir, split_times=split_times)
    return


def split_wav(in_dir, key, target_dir, split_times):
    '''
    Split wav at given split times/points
    Convert stereo to mono
    Export results
    '''
    wav_path = os.path.join(in_dir, f'{key}.wav')

    for count, split in enumerate(split_times):
        new_wav = AudioSegment.from_wav(wav_path)
        new_wav = new_wav[split[0]*1e3:split[1]*1e3]

        part = str(count).zfill(3)
        new_wav.export(os.path.join(target_dir, 'wav_mono', f'{key}_part_{part}.wav'), format='wav')        
    return


def split_annotations(target_dir, key, annotations, split_times):
    '''
    Split annotation csv at given split times
    Export results to target_dir
    '''
    start_key = 'MFA_ADJ_START'
    end_key = 'MFA_ADJ_END'

    for count, split in enumerate(split_times):
        # copy original annotations
        tmp = annotations[(annotations[end_key] >= split[0]) & (annotations[start_key] <= split[1])].copy().reset_index(drop=True)

        # timestamp for first annotation needs to be changed according to split time
        first_index = tmp.index[0]
        tmp.at[first_index, 'MFA_ADJ_START'] = split[0]
        tmp.loc[tmp['SYLLABLE_POS'] == tmp.at[first_index, 'SYLLABLE_POS'], 'SYLLABLE_START'] = split[0]

        # timestamp for last annotation needs to be changed according to split time
        last_index = tmp.index[-1]
        tmp.at[last_index, 'MFA_ADJ_END'] = split[1]
        tmp.loc[tmp['SYLLABLE_POS'] == tmp.at[last_index, 'SYLLABLE_POS'], 'SYLLABLE_END'] = split[1]

        # new annotations must start at time=0
        tmp['MFA_ADJ_START'] -= split[0]
        tmp['MFA_ADJ_END'] -= split[0]
        tmp['SYLLABLE_START'] -= split[0]
        tmp['SYLLABLE_END'] -= split[0]
        tmp['SYLLABLE_POS'] -= tmp.at[first_index, 'SYLLABLE_POS']-1
        
        # TODO syllable- or MFA-durations can be negative now

        # export
        part = str(count).zfill(3)
        tmp.to_csv(os.path.join(target_dir, 'csv', f'{key}_part_{part}.csv'), index=False)
    return


def find_split_times(annotations, max_unvoiced_time=0.9, max_length=10) -> list:
    '''
    Find split times for snippeting given a maximum length
    The preferred candidate for a split is the half time of an unvoiced part.
    max_unvoiced_time controls how long pauses at the beginning and end of a snippet can be.
    This is a faster algorithm than utils.find_optimal_split_2 but it fails sometimes
    '''
    annotations = annotations[~annotations.VOICED].reset_index()
    start_key = 'MFA_ADJ_START'
    end_key = 'MFA_ADJ_END'
    durations = annotations[end_key] - annotations[start_key]

    # list of tuples (start, end)
    split_times = []
    last_syllable = 0
    current_syllable = 0

    first_unvoiced_duration = durations.iloc[0]
    start_unvoiced = min(0.5 * first_unvoiced_duration, first_unvoiced_duration - max_unvoiced_time) 

    for row in annotations.index:
        unvoiced_time = min(0.5 * durations.iloc[row], max_unvoiced_time)
        end_candidate = annotations.loc[row, start_key] + unvoiced_time
            
        current_syllable = annotations.loc[row, 'SYLLABLE_POS']
        if current_syllable != last_syllable:
            if end_candidate - start_unvoiced > max_length:
                # if last_syllable + 1 != current_syllable: MFA could not find the syllables in between 
                # print(last_syllable, current_syllable)
                assert end_unvoiced - start_unvoiced > 0, 'Maximum snippet length too short.'
                split_times.append((start_unvoiced, end_unvoiced))
                start_unvoiced = start_candidate
                
            start_candidate = annotations.loc[row, end_key] - unvoiced_time
            end_unvoiced = end_candidate
            last_syllable = current_syllable

    last_unvoiced = annotations.iloc[-1]
    end_unvoiced = last_unvoiced[start_key] + min(0.5 * durations.iloc[-1], max_unvoiced_time)
    split_times.append((start_unvoiced, end_unvoiced))

    return split_times


def add_duration_labels(csv_dir, mel_dir, key):
    """
    Add duration labels that sum to the exact mel length
    """
    csv = pd.read_csv(os.path.join(csv_dir, f'{key}.csv'))
    with open(os.path.join(mel_dir, f'csd-mel-{key}.pkl'), 'rb') as f:
        mel_data = pickle.load(f)    
    mel_length = len(mel_data['mel'])

    # TODO some end times are smaller than the next one
    #assert np.all(csv['MFA_ADJ_END'].to_numpy()[:-1] <= csv['MFA_ADJ_END'].to_numpy()[1:]), f'{key}: End times are not ordered'

    # If the last note is completely out of mel length, remove it
    while csv['MFA_ADJ_END'].iloc[-2] * (hp.sampling_rate / hp.hop_length) > mel_length:
        csv.drop(len(csv)-1, axis=0, inplace=True)
    
    # Adjust the last note to match mel_length
    csv.loc[len(csv)-1, 'MFA_ADJ_END'] = mel_length * (hp.hop_length / hp.sampling_rate)

    # Get the cumsum durations by taking the time from the start of the file to the end of the note
    # This minimizes rounding errors
    durations = csv['MFA_ADJ_END'].to_numpy()
    # Convert to mel-frames
    durations = (durations * (hp.sampling_rate / hp.hop_length)).round().astype(int)

    # No negative or zero durations
    while np.any(durations[:-1] >= durations[1:]):
        durations[np.where(durations[:-1] >= durations[1:])[0]] -= 1
    if durations[0] <= 0:
        durations[0] = 1
    
    # Convert cumsum to individual durations
    durations[1:] -= durations[:-1].copy()
    
    # In theory there should be no error left
    assert np.all(durations >= 1), f'{key}: Negative duration encountered'
    assert abs(mel_length - sum(durations)) <= 1, f'{key}: Mismatch between mel length {mel_length} and durations {sum(durations)}'
    if mel_length != sum(durations):
        durations[-1] += mel_length-sum(durations)
    
    csv['DURATION_FRAMES'] = durations
    
    frame_indices = [(sum(durations[:i]), sum(durations[:i+1])) for i in range(len(durations))]

    # Syllable durations can be converted cheaper, doesn't need to be perfectly accurate
    durations = (csv['SYLLABLE_END'] - csv['SYLLABLE_START']).to_numpy()
    durations = (durations * (hp.sampling_rate / hp.hop_length)).round().astype(int)
    durations[durations <= 0] = 1
    csv['SYLLABLE_DURATION_FRAMES'] = durations

    # Annotate phoneme level information for sound features
    def log(x, epsilon=1):  # F0 information below epsilon threshold doesn't make sense
        x[x < epsilon] = epsilon
        return np.log(x)

    csv['INTENSITY'] = [mel_data['intensity'][start:end].mean() for start, end in frame_indices]
    csv['F0'] = [mel_data['f0'][start:end].mean() for start, end in frame_indices]
    csv['F0_STD'] = [mel_data['f0'][start:end].std() for start, end in frame_indices]
    csv['LOG_F0'] = [log(mel_data['f0'][start:end]).mean() for start, end in frame_indices]
    csv['LOG_F0_STD'] = [log(mel_data['f0'][start:end]).std() for start, end in frame_indices]
    csv['HNR'] = [mel_data['hnr'][start:end].mean() for start, end in frame_indices]
    csv['VOICED_FRAC'] = [mel_data['voiced'][start:end].mean() for start, end in frame_indices]
    csv['VIBRATO'] = csv['LOG_F0_STD']**2 * csv['HNR'] * (csv['VOICED_FRAC'] > 0.8) * 0.05



    csv.to_csv(os.path.join(csv_dir, f'{key}.csv'), index=False)
    return csv

def remove_phoneme_stress(csv_dir, key, stressed_phoneme_dict, unstressed_phoneme_dict):
    csv = pd.read_csv(os.path.join(csv_dir, f'{key}.csv'))
    stressed_phoneme_dict = {v:k for k,v in stressed_phoneme_dict.items()}

    csv['PHONEME'] = csv['PHONEME'].apply(lambda x: unstressed_phoneme_dict[stressed_phoneme_dict[x].replace('0', '').replace('1', '').replace('2', '')])
    csv['SYLLABLE'] = csv['SYLLABLE'].apply(lambda x: x.replace('0', '').replace('1', '').replace('2', ''))
    csv.to_csv(os.path.join(csv_dir, f'{key}.csv'), index=False)



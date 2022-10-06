import librosa
import numpy as np
import pandas as pd
import torch
import scipy.interpolate
import pickle

if __package__ is None or __package__ == '':
    import hparams as hp
else:
    from . import hparams as hp

def get_mel(wav_path):
    wave_audio, _ = librosa.load(wav_path, sr=hp.sampling_rate, mono=True)
    return audio_to_mel(wave_audio)

def audio_to_mel(wave_audio):
    mel = librosa.feature.melspectrogram(y=wave_audio,
                                         sr=hp.sampling_rate,
                                         n_fft=hp.win_length,
                                         hop_length=hp.hop_length,
                                         center=hp.center_mels,
                                         pad_mode='reflect',
                                         n_mels=hp.num_mels,
                                         fmin=hp.mel_fmin,
                                         fmax=hp.mel_fmax)

    return mel

def mel_to_audio(mel, denorm=False):
    mel = np.array(mel)
    if denorm:
        mel = denorm_mel(mel)
    assert mel.shape[0] == hp.num_mels
    wave = librosa.feature.inverse.mel_to_audio(M=mel,
                                                sr=hp.sampling_rate,
                                                n_fft=hp.win_length,
                                                hop_length=hp.hop_length,
                                                center=hp.center_mels,
                                                pad_mode='reflect',
                                                #n_mels=hp.num_mels,
                                                fmin=hp.mel_fmin,
                                                fmax=hp.mel_fmax)
    return wave


mel_norm_factor = 10
mel_norm_offset = 3
# TODO this is ugly
#with open(hp.mel_norm_file, 'rb') as f:
#    data = pickle.load(f)
#    mel_norm_factor = data['factor']
#    mel_norm_offset = data['offset']
#    del data

def norm_mel(mel):
    
    if len(mel.shape) == 3:
        return np.stack([norm_mel(mel[i]) for i in range(mel.shape[0])])

    # To log scale
    mel = librosa.power_to_db(mel)
    # Transpose
    mel = mel.transpose()
    # Normalize
    mel = mel / mel_norm_factor + mel_norm_offset


    return mel

# Invert norm_mel
def denorm_mel(mel):
    if len(mel.shape) == 3:
        return np.stack([denorm_mel(mel[i]) for i in range(mel.shape[0])])

    mel = (mel - mel_norm_offset) * mel_norm_factor
    mel = mel.transpose()
    mel = librosa.db_to_power(mel)

    return mel

def load_note_conversion_table(file):
    table = pd.read_csv(file, header=None)
    mapper = np.ones(max(table[0])+1)
    mapper[table[0]] = table[3]
    return norm_note(mapper)

def bin_duration(dur):
    if isinstance(dur, (np.ndarray, np.generic)):
        dur = dur.astype(float)
    elif torch.is_tensor(dur):
        dur = dur.double()
    else:
        raise "Not Implemented"

    # Durations to [0,1]
    dur = (dur - hp.bin_durations_min) / hp.bin_durations_max
    # Durations to bin range, then round to the nearest int bin and clip
    dur = (dur * hp.bin_durations_count + 1).round().clip(1, hp.bin_durations_count)

    if isinstance(dur, (np.ndarray, np.generic)):
        dur = dur.astype(int)
    elif torch.is_tensor(dur):
        dur = dur.long()

    return dur

def debin_duration(dur):
    if isinstance(dur, (np.ndarray, np.generic)):
        dur = dur.astype(float)
    elif torch.is_tensor(dur):
        dur = dur.double()
    else:
        raise "Not Implemented"

    # Durations to [0,1]
    dur = (dur - 1) / hp.bin_durations_count
    # Durations to bin range, then round to the nearest int bin and clip
    dur = (dur * (hp.bin_durations_max - hp.bin_durations_min) + hp.bin_durations_min).round()

    if isinstance(dur, (np.ndarray, np.generic)):
        dur = dur.astype(int)
    elif torch.is_tensor(dur):
        dur = dur.long()

    return dur


def norm_note(note):
    note = np.array(note)
    note[note <= 0] = 1
    return np.log(note)


def denorm_note(note):
    retval = np.exp(note)
    retval[note == 0] = 0
    return retval



def norm_hnr(hnr):
    return (hnr - hp.hnr_norm_min) / (hp.hnr_norm_max - hp.hnr_norm_min)

def denorm_hnr(hnr):
    return hnr * (hp.hnr_norm_max - hp.hnr_norm_min) + hp.hnr_norm_min

def norm_intensity(intensity):
    return (intensity - hp.intensity_norm_min) / (hp.intensity_norm_max - hp.intensity_norm_min)

def denorm_intensity(intensity):
    return intensity * (hp.intensity_norm_max - hp.intensity_norm_min) + hp.intensity_norm_min

def interpolate_f0(f0, voiced):
    assert f0.shape == voiced.shape
    if len(f0.shape) == 2:
        return np.stack([interpolate_f0(f0[i], voiced[i]) for i in range(f0.shape[0])])
    assert len(f0.shape) == 1

    voiced = voiced.astype(bool)

    indices = np.arange(len(f0))
    interp = scipy.interpolate.interp1d(indices[voiced], f0[voiced], kind='linear', bounds_error=False, fill_value='extrapolate')
    f0[~voiced] = interp(indices[~voiced])
    return f0

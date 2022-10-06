import pickle
import os
import pretty_midi
import warnings
import numpy as np
import torch
import math

if __package__ is None or __package__ == '' or __package__ == 'data':
    import hparams as hp
    import audio
    import utils
else:
    from .. import hparams as hp
    from .. import audio
    from .. import utils

hop_length_factor = hp.sampling_rate / hp.hop_length


def find_in_dict(syll, phoneme_dict):
    '''
    Finds the phonemes from the syllable in the phoneme dict
    If the split doesn't work, try to fix the split manually
    '''
    output_chars = []
    for char in syll.split('_'):
        if char in phoneme_dict:
            output_chars.append(char)
            continue

        last_match = 0
        for idx in range(1, len(char)+1):
            if char[last_match:idx] in phoneme_dict:
                output_chars.append(char[last_match:idx])
                last_match = idx
        
        if last_match < len(char):
            warnings.warn(f'Phoneme not in dict: {char[last_match:]}, using <nosound> instead')
            output_chars.append('<nosound>')

    return [phoneme_dict[char] for char in output_chars]

def load_textnotegen_format(textnotegen_dir, split_mode='auto', pitch_shift=0):
    #with open(os.path.join('data', 'csd_phoneme_dict.pkl'), 'rb') as f:
    #    phoneme_dict = pickle.load(f)
    #with open(os.path.join('data', 'mfa_csd_dict.txt'), 'r') as f:
    #    mfa_csd_dict = f.read().split('\n')
    #mfa_csd_dict = {l.split(' ')[1].strip().replace('1', ''):l.split(' ')[0].strip() for l in mfa_csd_dict if len(l)}
    #mfa_csd_dict['<nosound>'] = '<nosound>'
    with open(hp.phoneme_dict, 'rb') as f:
        phoneme_dict = pickle.load(f)
    phoneme_dict_inv = {v:k.replace('1', '').replace('2', '') for k,v in phoneme_dict.items()}

    # Read text and note information
    with open(os.path.join(textnotegen_dir, 'txt_punctuation.txt'), 'r') as f:
        text = f.read()
    text = [t for t in text.strip().split(' ')]
    # There is a <punctuation> where the text could be split
    split_locations = np.array([i for i,c in enumerate(text[:-1]) if c == '<punctuation>'])
    split_locations = split_locations - np.arange(len(split_locations)) - 1
    assert np.all(split_locations >= 0)
    assert np.all(split_locations < len(text))
    
    text = [t for t in text if t != '<punctuation>' and len(t)]
    # Convert the IPA phonemes into phoneme dict indices
    text = [find_in_dict(syll, phoneme_dict) for syll in text]
    #text = [[phoneme_dict[char] for char in syll] for syll in text] # Now its list of lists of ids

    pm = pretty_midi.PrettyMIDI(os.path.join(textnotegen_dir, 'melody.mid'))
    notes = list(pm.instruments[0].notes)
 
    # The lengths of notes and text must match
    if len(notes) > len(text):
        warnings.warn('Text length and note length do not match, cropping')
        notes = notes[0:len(text)]
    if len(text) > len(notes):
        warnings.warn('Text length and note length do not match, cropping')
        text = text[0:len(notes)]
        split_locations = split_locations[split_locations < len(notes)]
    # Insert <nosound> between midi notes
    idx = 1
    nosound_locations = []
    while idx < len(text):
        if notes[idx].start > notes[idx-1].end:
            notes.insert(idx, pretty_midi.Note(0, 1, notes[idx-1].end, notes[idx].start))
            text.insert(idx, [phoneme_dict['<nosound>']])
            split_locations = [s+1 if s >= idx else s for s in split_locations]
            nosound_locations.append(idx)
        idx += 1
    nosound_locations = np.array(nosound_locations)
    pitches = [max(n.pitch + pitch_shift, 0) for n in notes]
    syllable_durations_raw = np.array([int(round((n.end - n.start) * hop_length_factor)) for n in notes])
    if hp.bin_durations:
        syllable_durations = audio.bin_duration(syllable_durations_raw)
    else:
        syllable_durations = syllable_durations_raw.copy()

    # Find the optimal split
    max_split_length = hp.max_snippet_length * (hp.sampling_rate/hp.hop_length)
    if split_mode == 'auto':
        split_locations = utils.find_optimal_split(syllable_durations_raw.cumsum(), split_locations, max_split_length)
        split_locations = np.concatenate([[0], split_locations, [len(text)-1]])
        split_locations = np.sort(split_locations)
    elif split_mode == 'none':
        split_locations = np.array([0, len(text)-1])
    elif split_mode == 'all':
        split_locations = np.concatenate([[0], split_locations, [len(text)-1]])
        split_locations = np.unique(split_locations)
        split_locations = np.sort(split_locations)
        #for idx in range(1, len(split_locations)):
    split_locations = split_locations.astype(int) 

    data_out = []
    split_locations[0] -= 1
    split_locations += 1
    
    # Check for to long split lenghts
    while True:
        split_lengths = np.diff(np.concatenate([[0], syllable_durations_raw.cumsum()])[split_locations])
        if split_mode == 'none' or not np.any(split_lengths > max_split_length):
            break
        first_violation = np.argmax(split_lengths > max_split_length)
        # find the last nosound location in the violating split and add it to the split locations array
        mask = (nosound_locations > split_locations[first_violation]) & (nosound_locations < split_locations[first_violation+1])
        matching_nosounds = np.sort(nosound_locations[mask])
        if len(matching_nosounds) == 0:
            if split_locations[first_violation] == split_locations[first_violation+1]-1:
                warnings.warn(f'Snippet length {split_lengths[first_violation]} exceeds maximum split, but the note is so long that no split can be found, the synthesis will be cropped')
                break
            warnings.warn(f'Snippet length {split_lengths[first_violation]} exceeds maximum split and neither a punctuation nor a split mark found in the long snippet, cutting in half')
            split_locations = np.concatenate([[(split_locations[first_violation] + split_locations[first_violation+1]) // 2], split_locations])
        else:
            warnings.warn(f'Snippet length {split_lengths[first_violation]} exceeds maximum split, adding a nosound split')
            split_locations = np.concatenate([[matching_nosounds[len(matching_nosounds)//2]], split_locations])
        split_locations = np.sort(split_locations)
  

    for start, end in zip(split_locations[:-1], split_locations[1:]):
        # Expand the notes to a flattened version of the text
        text_out = []
        syll_out = []
        pitches_out = []
        syllable_durations_out = []
        syllable_durations_raw_out = []
        syllable_pos_out = []

        syllable_pos_offset = 1 #Syllable positions should start at 1

        # If there is a <nosound> before the split, add half of it here
        if start > 0 and text[start-1] == [1]:
            text_out.append(1)
            pitches_out.append(pitches[start-1])
            syllable_durations_out.append(int(math.floor(syllable_durations[start-1]/2)))
            syllable_durations_raw_out.append(int(math.floor(syllable_durations_raw[start-1]/2)))
            syllable_pos_out.append(1)
            syllable_pos_offset = 2
            syll_out.append(phoneme_dict_inv[1])

        for idx in range(start, end):
            syl = text[idx]
            syll_out.append("_".join([phoneme_dict_inv[char] for char in syl]))
            for char in syl:
                text_out.append(char)
                pitches_out.append(pitches[idx])
                syllable_durations_out.append(syllable_durations[idx])
                syllable_durations_raw_out.append(syllable_durations_raw[idx])
                syllable_pos_out.append(idx + syllable_pos_offset - start)

        # If the last item was a <nosound> and it's not the end of the snippet, cut it in half as the next split will take the other half
        if end < max(split_locations) and text[end-1] == [1]:
            syllable_durations_out[-1] = int(math.ceil(syllable_durations_out[-1]/2))
            syllable_durations_raw_out[-1] = int(math.ceil(syllable_durations_raw_out[-1]/2))


        data_out.append({
            'text': np.array(text_out), 
            'note': np.array(pitches_out), 
            'syllable_duration': np.array(syllable_durations_out),
            'syllable_duration_raw': np.array(syllable_durations_raw_out),
            'syllable_pos': np.array(syllable_pos_out),
            'energy': np.zeros(len(text_out)),
            'syllables': ' '.join(syll_out)
        })
    return data_out

def fix_note(note):
    assert hp.note_vocab_size > 12, 'Too small note vocab, endless recursion'

    if note >= hp.note_vocab_size:
        warnings.warn(f'Too high note encountered: {note}')
        note = fix_note(note-12) # One octave down, check again
    if note < 1:
        warnings.warn(f'Too low note encountered: {note}')
        note = fix_note(note+12) # One octave up, check again
    return note

def fix_dur(dur):
    if dur >= hp.syllable_duration_vocab_size:
        warnings.warn(f'Too long duration encountered: {dur}')
        dur = hp.syllable_duration_vocab_size -1
    if dur <= 0:
        warnings.warn(f'Duration <= 0 encountered')
        dur = 1
    return dur

def fix_data(data):
    text = data['text']
    notes = data['note']
    syllable_durations = data['syllable_duration']
    syllable_durations_raw = data['syllable_duration_raw']
    syllable_pos = data['syllable_pos']
    energy = data['energy']
    
    assert len(text) == len(notes)
    assert len(text) == len(syllable_durations)
    assert len(text) == len(syllable_pos)
    assert len(text) == len(energy)
    assert len(text) == len(syllable_durations_raw)

    if len(text) >= hp.max_seq_len_txt:
        warnings.warn(f'Textnotegen output length ({len(text)}) greater than max seq len, cropping')
        text = text[0:hp.max_seq_len_txt]
        notes = notes[0:hp.max_seq_len_txt]
        syllable_durations = syllable_durations[0:hp.max_seq_len_txt]
        syllable_durations_raw = syllable_durations_raw[0:hp.max_seq_len_txt]
        syllable_pos = syllable_pos[0:hp.max_seq_len_txt]
        energy = energy[0:hp.max_seq_len_txt]

    notes = np.array([fix_note(n) for n in notes])
    syllable_durations = np.array([fix_dur(d) for d in syllable_durations])

    return {
        'text': text,
        'note': notes,
        'syllable_duration': syllable_durations,
        'syllable_duration_raw': syllable_durations_raw,
        'syllable_pos': syllable_pos,
        'energy': energy,
        'syllables': data['syllables']
    }




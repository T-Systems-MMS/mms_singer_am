import os
import shutil
import pandas as pd
import praatio.data_classes.interval_tier
import praatio.data_classes.textgrid
import praatio.utilities.constants
import warnings
import json
import argparse

from tqdm import tqdm
from pydub import AudioSegment


def _process_syllable(text):
    return text.replace('_', ' ')
    

def gen_textgrid(dataset, output, key):
    note_sequence = pd.read_csv(os.path.join(dataset, f'{key}.csv'))
    global_start = note_sequence['start'].min()
    global_end = note_sequence['end'].max()

    for i in range(len(note_sequence)-1):
        if note_sequence.iloc[i]['end'] > note_sequence.iloc[i+1]['start']:
            #warnings.warn(f'{key}: overlapping intervals at iloc {i}')
            note_sequence.loc[i,'end'] = note_sequence.iloc[i+1]['start']

    syllables = [praatio.utilities.constants.Interval(start, end, _process_syllable(syllable)) for (start, end, _, syllable) in note_sequence.values]
    syllables_tier = praatio.data_classes.interval_tier.IntervalTier(name='1', entries=syllables, minT=global_start, maxT=global_end)
    
    tgfile = praatio.data_classes.textgrid.Textgrid(global_start, global_end)
    tgfile.addTier(syllables_tier, 1)
    tgfile.save(fn=os.path.join(output, f'{key}.TextGrid'), format='short_textgrid', includeBlankSpaces=True)


def convert_to_mono(in_dir, target_dir, key):
    new_wav = AudioSegment.from_wav(os.path.join(in_dir, f'{key}.wav'))
    new_wav = new_wav.set_channels(1)
    new_wav.export(os.path.join(target_dir, f'{key}.wav'), format='wav')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, default=os.path.join('CSD', 'english'))
    parser.add_argument('--target_dir', type=str, default='mfa_csd_corpus')
    args = parser.parse_args()

    in_dir = args.in_dir
    target_dir = args.target_dir

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    keys = [key.partition('.')[0] for key in os.listdir(os.path.join(args.in_dir, 'csv'))]
    keys.sort()
    print(f'Processing {len(keys)} songs to MFA corpus format')
    # enhance existing labels (csv format)
    # create short snippets (csv+wav) from the original CSD
    for key in tqdm(keys):
        convert_to_mono(os.path.join(in_dir, 'wav'), target_dir, key)
        gen_textgrid(os.path.join(in_dir, 'csv'), target_dir, key)

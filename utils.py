import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import math
import numpy as np
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import pandas as pd
import pickle
import itertools
import warnings
import pretty_midi
from mcts import mcts

if __package__ is None or __package__ == '':
    import hparams
    import audio
else:
    from . import hparams
    from . import audio

hop_length_frames = 1e-3 * hparams.sampling_rate / hparams.hop_length
hop_length_sec = hparams.hop_length / hparams.sampling_rate

def sec_to_melframes(sec):
    return round(sec * hop_length_frames)

def melframes_to_sec(melframes):
    return melframes*hop_length_sec

def spec_to_img(mel, f0=None, voiced=None):
    mel = np.array(mel).transpose()

    my_dpi=300
    fig, ax = plt.subplots(figsize=(mel.shape[1]/my_dpi, mel.shape[0]/my_dpi), dpi=my_dpi)
    canvas = FigureCanvas(fig)

    #S_dB = librosa.power_to_db(mel, ref=np.max)

    if f0 is not None and voiced is not None:
        f0 = np.array(f0)
        voiced = np.array(voiced)
        
        # Denorm and clip f0
        f0 = audio.denorm_note(f0)

        # Apply Sigmoid if it's a float value
        if np.issubdtype(voiced.dtype, np.floating):
            voiced = (1 / (1 + np.exp(-voiced))).round().astype(int)
        
        # Convert to mel scale
        mel_frequencies = librosa.mel_frequencies(n_mels=hparams.num_mels, fmax=hparams.mel_fmax, fmin=hparams.mel_fmin)
        idx = np.searchsorted(mel_frequencies, f0)
        
        x = np.array(list(range(len(idx))))

        idx = idx[voiced == 1]
        x = x[voiced == 1]
        
        x = x[(idx < 80) & (idx >= 0)]
        idx = idx[(idx < 80) & (idx >= 0)]


        mel_max = np.min(mel)
        for (x, y) in zip(x, idx):
            mel[y,x] = mel_max

    img = librosa.display.specshow(mel, x_axis='frames',
                                   y_axis='mel', sr=hparams.sampling_rate,
                                   fmin=hparams.mel_fmin, fmax=hparams.mel_fmax, ax=ax)

    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    

        

    canvas.draw()
    buf = canvas.buffer_rgba()
    # convert to a NumPy array
    X = np.asarray(buf)
    plt.close(fig)


    return X[:,:,0:3] # discart alpha information


def process_text(train_text_path):
    with open(train_text_path, "r", encoding="utf-8") as f:
        txt = []
        for line in f.readlines():
            txt.append(line)

        return txt


def get_param_num(model, trainable_only=False):
    if trainable_only:
        return sum(param.numel() for param in model.parameters() if param.requires_grad)
    
    return sum(param.numel() for param in model.parameters())

def get_mask_from_lengths(lengths, max_len=None):
    if max_len == None:
        max_len = torch.max(lengths)

    ids = torch.arange(0, max_len, dtype=torch.long, device=lengths.device)
    mask = (ids < lengths.unsqueeze(1)).bool()

    return mask

def pad_1D(inputs, PAD=0):

    def pad_data(x, length, PAD):
        x_padded = np.pad(x, (0, length - x.shape[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_1D_tensor(inputs, PAD=0):

    def pad_data(x, length, PAD):
        x_padded = F.pad(x, (0, length - x.shape[0]))
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = torch.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):

    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(x, (0, max_len - np.shape(x)[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad_2D_tensor(inputs, maxlen=None):

    def pad(x, max_len):
        if x.size(0) > max_len:
            raise ValueError("not max_len")

        s = x.size(1)
        x_padded = F.pad(x, (0, 0, 0, max_len-x.size(0)))
        return x_padded[:, :s]

    if maxlen:
        output = torch.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(x.size(0) for x in inputs)
        output = torch.stack([pad(x, max_len) for x in inputs])

    return output


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        out_list = list()
        max_len = mel_max_length
        for i, batch in enumerate(input_ele):
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len-batch.size(0)), "constant", 0.0)
            out_list.append(one_batch_padded)
        out_padded = torch.stack(out_list)
        return out_padded
    else:
        out_list = list()
        max_len = max([input_ele[i].size(0)for i in range(len(input_ele))])

        for i, batch in enumerate(input_ele):
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len-batch.size(0)), "constant", 0.0)
            out_list.append(one_batch_padded)
        out_padded = torch.stack(out_list)
        return out_padded

def processed_to_df(data, phoneme_dict=None, join_syllables=False):
    if phoneme_dict is None:
        with open(hparams.phoneme_dict, 'rb') as f:
            phoneme_dict = pickle.load(f)
    # Reverse phoneme dict
    phoneme_dict = {v:k for k,v in phoneme_dict.items()}
    
    df = pd.DataFrame()
    df['note'] = data['note']
    df['text_id'] = data['text']
    df['text'] = df['text_id'].apply(lambda x: phoneme_dict[x])
    df['syllable_dur'] = data['syllable_duration']
    df['syllable_pos'] = data['syllable_pos']
    if 'duration' in data:
        df['duration'] = data['duration']

    if join_syllables:
        df_joined = pd.DataFrame()
        df_joined['text'] = list(df.groupby('syllable_pos')['text'].apply(lambda x: "_".join([*x])))
        df_joined['note'] = list(df.groupby('syllable_pos')['note'].first())
        df_joined['syllable_dur'] = list(df.groupby('syllable_pos')['syllable_dur'].first())

        df = df_joined
        df['end'] = df['syllable_dur'].cumsum() / hparams.sampling_rate * hparams.hop_length
        df['start'] = ([0] + list(df['end']))[:len(df)]

    return df


def csv_to_midi(csv_file, midi_file=None):
    """
    Converts a csv file in CSD or CSD_processed format to midi
    """
    csv = pd.read_csv(csv_file)
    if 'PITCH' in csv.columns:
        pitches = csv.groupby('SYLLABLE_POS').first()['PITCH'].to_numpy()
        starts = csv.groupby('SYLLABLE_POS').first()['SYLLABLE_START'].to_numpy()
        ends = csv.groupby('SYLLABLE_POS').first()['SYLLABLE_END'].to_numpy()
        texts = ["_".join(syl) for syl in csv_to_txt(csv_file)]
        assert len(texts) == len(starts)
    else:
        pitches = csv['pitch'].to_numpy()
        starts = csv['start'].to_numpy()
        ends = csv['end'].to_numpy()
        texts = list(csv['syllable'])


    midi_out = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=55) # Program is the type of instrument
    for (pitch, start, end) in zip(pitches, starts, ends):
        instrument.notes.append(pretty_midi.Note(
            velocity=100,
            pitch=pitch,
            start=start,
            end=end))
    midi_out.instruments.append(instrument)
    
    for (start, text) in zip(starts, texts):
        midi_out.lyrics.append(pretty_midi.Lyric(text, start))

    if midi_file is not None:
        midi_out.write(str(midi_file))
    return midi_out

def csv_to_txt(csv_file, txt_file=None, phoneme_dict=None):
    """
    Converts a csv file with a PHONEME id column to a txt file in textnotegen format
    """
    csv = pd.read_csv(csv_file)
    with open(hparams.phoneme_dict, 'rb') as f:
        phoneme_dict = pickle.load(f)
    phoneme_dict = {v:k for k,v in phoneme_dict.items()}
    phoneme_dict[1] = '<nosound>'

    sylpos = csv.loc[csv.index[0], 'SYLLABLE_POS']
    texts = []
    syllable = []
    for row in csv.index:
        if csv.loc[row, 'SYLLABLE_POS'] != sylpos:
            sylpos = csv.loc[row, 'SYLLABLE_POS']
            texts.append(syllable)
            syllable = []
        syllable.append(phoneme_dict[csv.loc[row, 'PHONEME']])
    texts.append(syllable)
    syllable = []

    joined = " ".join(["_".join(syl) for syl in texts])
    if txt_file is not None:
        with open(txt_file, 'w') as f:
            f.write(joined)
    return texts

def csv_folder_to_textnotegen(in_folder, out_folder):
    files = [x for x in os.listdir(in_folder) if x.endswith('.csv')]
    for file in files:
        res_folder = os.path.join(out_folder, file.partition('.')[0])
        os.makedirs(res_folder, exist_ok=True)
        csv_to_txt(os.path.join(in_folder, file), os.path.join(res_folder, 'txt_punctuation.txt'))
        csv_to_midi(os.path.join(in_folder, file), os.path.join(res_folder, 'melody.mid'))
        shutil.copy(os.path.join(res_folder, 'txt_punctuation.txt'), os.path.join(res_folder, 'lyrics.txt'))

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

# TODO this function has performance optinization potential
def find_optimal_split(timestamps, split_locations, max_split_length, fallback_locations=None):
    """
    This method filters the possible split locations to find the selection of split locations that
    maximizes average split length with the constraint of not exceeding max_split_length
    Runs in exponential time with regard to split_locations, make sure len(split_locations) stays small
    """
    timestamps = np.array(timestamps)
    assert len(timestamps)
    assert len(split_locations) < 20, 'Too many possible split locations'
    possible_splits = []
    max_lengths = []
    mean_lengths = []

    for split in powerset(split_locations):
        # Always append the end as there is always a cut there
        split = np.sort(np.array(list(split)).astype(int))
        diffs = np.diff(np.concatenate([[0], timestamps[split], timestamps[[-1]]]))
        possible_splits.append(split)
        max_lengths.append(diffs.max())
        mean_lengths.append(diffs.mean() ** 2)

    max_lengths = np.array(max_lengths)
    mean_lengths = np.array(mean_lengths)
    if max_lengths.min() > max_split_length:
        if fallback_locations is not None:
            warnings.warn("Could not find a split that satisfies the max_split_length constraint, retrying with extra split locations from <nosounds>")
            fallback_locations = np.concatenate((np.array(split_locations), np.array(fallback_locations)))
            return find_optimal_split(timestamps, fallback_locations, max_split_length, None)
        warnings.warn("Could not find a split that satisfies the max_split_length constraint, using the split that violates the condition the least")
        max_split_length = max_lengths.min()

    # Zero all constraint violations, then take argmax
    mean_lengths *= max_lengths <= max_split_length
    return possible_splits[mean_lengths.argmax()]

class SplitLocMCTS():
    def __init__(self, split_points, min_split_length, max_split_length, splits=[0]):
        self.split_points = split_points
        self.splits = splits # Array of split point indices that are used
        self.min_split_length = min_split_length
        self.max_split_length = max_split_length

    def getCurrentPlayer(self):
        return 1
    
    def getPossibleActions(self):
        possible_actions = []
        for i in range(self.splits[-1]+1, len(self.split_points)):
            if (self.split_points[i][0] - self.split_points[self.splits[-1]][1]) < self.min_split_length:
                continue
            if (self.split_points[i][0] - self.split_points[self.splits[-1]][1]) > self.max_split_length:
                break
            possible_actions.append(i)
        return possible_actions

    def hasPossibleAction(self):
        for i in range(self.splits[-1]+1, len(self.split_points)):
            if (self.split_points[i][0] - self.split_points[self.splits[-1]][1]) < self.min_split_length:
                continue
            if (self.split_points[i][0] - self.split_points[self.splits[-1]][1]) > self.max_split_length:
                break
            return True
        return False

    def takeAction(self, action):
        return SplitLocMCTS(self.split_points, self.min_split_length, self.max_split_length, self.splits + [action])

    def isTerminal(self):
        return self.splits[-1] >= len(self.split_points)-1 or self.split_points[self.splits[-1]][1] > self.split_points[-1][0]-self.min_split_length or not self.hasPossibleAction()

    def sumTime(self, splitidx_start, splitidx_end):
        return sum([self.split_points[i+1][0] - self.split_points[i][1] for i in range(splitidx_start, splitidx_end)])

    def splitLength(self, splitidx_start, splitidx_end):
        return self.split_points[splitidx_end][0] - self.split_points[splitidx_start][1]

    def getReward(self):
        if self.splits[-1] < len(self.split_points)-1:
            return 0 # min_split_length condition not fulfilled
        return np.mean([self.sumTime(self.splits[i], self.splits[i+1]) ** 4 for i in range(len(self.splits)-1)])


def find_optimal_split_2(timestamps_mfa, timestamps_syl, voiced, max_split_length, max_unvoiced_length, min_split_length=0):
    """
    This is a more sophisticated version of find_optimal_split which tries to achieve several things
    * maximize average (squared) split length
    with the constraint of
    * split only in mfa-unvoiced locations
    * crop long unvoiced sections out of the file
    * split such that all the splits are shorter than max_split_length
    * split where both the mfa and syl timestamps are unvoiced

    If constraints can not be satisfied, they shall be given up in inverse order of listing
    Returns an array of tuples with split start and end times
    Assumes that in the input array, no unvoiced section follows after an unvoiced section
    Runs as a tree search
    """

    assert len(timestamps_mfa) == len(timestamps_syl)
    assert len(timestamps_mfa) == len(voiced)
    assert len(timestamps_mfa[0]) == 2
    assert len(timestamps_syl[0]) == 2
    assert np.any(voiced)
    assert np.all([voiced[i+1] or voiced[i] for i in range(len(voiced)-1)])
    n = len(voiced)

    # Intersect timestamps_mfa and timestamps_syl
    intersect_times = [(i, max(timestamps_mfa[i][0], timestamps_syl[i][0]), min(timestamps_mfa[i][1], timestamps_syl[i][1])) for i in range(n)]
    
    # Find first and last voiced note
    first_split = 0
    if not voiced[0]:
        # Sample starts with an unvoiced section -> it starts at the end of the unvoiced
        first_split = max(0, min(timestamps_mfa[0][1], timestamps_syl[0][1]) - max_unvoiced_length*0.5)
        intersect_times = intersect_times[1:]

    last_split = max(timestamps_mfa[-1][1], timestamps_syl[-1][1])
    if not voiced[-1]:
        # Sample ends with an unvoiced section -> It ends at the beginning of the unvoiced
        last_split = min(last_split, max(timestamps_mfa[-1][0], timestamps_syl[-1][0]) + max_unvoiced_length*0.5)
        intersect_times = intersect_times[:-1]

    # Filter out unvoiced and negative length samples
    intersect_times = [(start, end) for (i, start, end) in intersect_times if not voiced[i] and start <= end]

    # Create split points from the intersect_times
    # Different start and ending for long unvoiced, same for short unvoiced
    def make_split_points(start, end):
        if end-start > max_unvoiced_length:
            return (start+max_unvoiced_length*0.5, end-max_unvoiced_length*0.5)
        else:
            return ((end+start)/2, (end+start)/2)
    split_points = [make_split_points(start, end) for (start, end) in intersect_times]
    split_points = [(None, first_split)] + split_points + [(last_split, None)]

    # Perform a MCTS to find the optimal split
    # Enumerating all options would be too costly (2**n complexity)
    # Also, exhaustive tree search turned out too costly (average branching factor ~15, average depth ~9) -> 15**9
    # Hence, MCTS
    splits = [0]
    end_reward = 0
    while True:
        mcts_start = SplitLocMCTS(split_points, min_split_length, max_split_length, splits)
        if mcts_start.isTerminal():
            end_reward = mcts_start.getReward()
            break
        searcher = mcts(timeLimit=1500)
        action = searcher.search(initialState=mcts_start)
        splits.append(action)

    if end_reward == 0:
        warnings.warn('Could not find a satisfactory split, relaxing conditions')
        if timestamps_mfa != timestamps_syl:
            return find_optimal_split_2(timestamps_mfa, timestamps_mfa, voiced, max_split_length, max_unvoiced_length, min_split_length)
        if min_split_length > 0:
            return find_optimal_split_2(timestamps_mfa, timestamps_mfa, voiced, max_split_length, max_unvoiced_length, 0)
        new_max_split_length = max([(split_points[i+1][0] - split_points[i][1]) for i in range(len(split_points)-1)])
        if new_max_split_length > max_split_length:
            return find_optimal_split_2(timestamps_mfa, timestamps_mfa, voiced, new_max_split_length, max_unvoiced_length, 0)

        raise 'Can not find a split'

    
    split_points = [p for (i, p) in enumerate(split_points) if i in splits]
    # Inverse the representation from (start, end) of unvoiced sections to (start, end) of voiced
    return [(split_points[i][1], split_points[i+1][0]) for i in range(len(split_points)-1)]

# Taken from https://discuss.pytorch.org/t/how-to-use-my-own-sampler-when-i-already-use-distributedsampler/62143/8
class DistributedWeightedSampler(torch.utils.data.Sampler):
    def __init__(self, weights, num_samples=None, num_replicas=None, rank=None, shuffle=True, replacement=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if num_samples is None:
            num_samples=len(weights)
        self.weights = weights / sum(weights)
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.len_dataset = num_samples
        self.num_samples = int(math.ceil(num_samples * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.replacement = replacement
        self.shuffle = shuffle


    def calculate_weights(self, targets):
        class_sample_count = torch.tensor(
            [(targets == t).sum() for t in torch.unique(targets, sorted=True)])
        weight = 1. / class_sample_count.double()
        samples_weight = torch.tensor([weight[t] for t in targets])
        return samples_weight

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(self.len_dataset, generator=g).tolist()
        else:
            indices = list(range(self.len_dataset))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(torch.multinomial(self.weights, self.num_samples, self.replacement).tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class WrappedModel(nn.Module):
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module # that I actually define.
    def forward(self, x):
        return self.module(x)

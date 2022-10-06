from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pickle
import torch
import audio
from tqdm import tqdm
import hparams as hp

class DiffDataset(Dataset):
  def __init__(self, gt_mel_path, fs_mel_path):
    gt_keys = os.listdir(gt_mel_path)
    fs_keys = os.listdir(fs_mel_path)
    assert len(gt_keys) == len(fs_keys)
    gt_keys.sort()
    fs_keys.sort()
    gt_keys = [os.path.join(gt_mel_path, f) for f in gt_keys]
    fs_keys = [os.path.join(fs_mel_path, f) for f in fs_keys]

    self.buffer = []
    self.weights = []
    for gt,fs in tqdm(zip(gt_keys, fs_keys), total=len(gt_keys)):
      with open(gt, 'rb') as f:
        gt_mel = pickle.load(f)['mel']
      with open(fs, 'rb') as f:
        data = pickle.load(f)
        fs_mel = data['mel']
        fs_dec_input = data['dec_input']
        sample_prop = data.get('sample_prop', np.array([1.0])).item()
 
      assert len(gt_mel) > 0
      assert len(gt_mel) == len(fs_mel)
      assert len(fs_dec_input) == len(fs_mel)
      assert gt_mel.shape[1] == hp.num_mels
      assert fs_mel.shape[1] == hp.num_mels
      assert fs_dec_input.shape[1] == hp.decoder_dim
      assert sample_prop > 0.0 and sample_prop <= 1.0

      #fs_mel = audio.norm_mel(fs_mel)

      self.buffer.append({
        'gt_mel': torch.tensor(gt_mel),
        'fs_mel': torch.tensor(fs_mel),
        'fs_dec_input': torch.tensor(fs_dec_input),
        'mel_pos': torch.arange(len(gt_mel))+1,
      })
      self.weights.append(sample_prop)

    self.weights = torch.tensor(self.weights)
  
  def __len__(self):
    return len(self.buffer)
  
  def __getitem__(self, idx):
    return self.buffer[idx]

  def get_weights(self):
    return self.weights
  
  def spec_min(self, gt=True):
    if gt:
      key = 'gt_mel'
    else:
      key = 'fs_mel'
      
    return np.stack([d[key].numpy().min(axis=0) for d in self.buffer]).min(axis=0)
  
  def spec_max(self, gt=True):
    if gt:
      key = 'gt_mel'
    else:
      key = 'fs_mel'
      
    return np.stack([d[key].numpy().max(axis=0) for d in self.buffer]).max(axis=0)

def collate_fn_pad(batch):
    keys = ['fs_mel', 'gt_mel', 'fs_dec_input', 'mel_pos']
    output = {key:torch.nn.utils.rnn.pad_sequence([d[key] for d in batch], batch_first=True) for key in keys}
    return output

if __name__ == "__main__":
  import hparams as hp
  test_ds = DiffDataset(hp.mel_ground_truth_test, hp.mel_fastspeech_test)
  train_ds = DiffDataset(hp.mel_ground_truth, hp.mel_fastspeech)

  print(f'Number of mels: {len(test_ds)}')
  print(f'GT Mel Range: {test_ds.spec_min(gt=True).min()} - {test_ds.spec_max(gt=True).max()}')
  print(f'FS Mel Range: {test_ds.spec_min(gt=False).min()} - {test_ds.spec_max(gt=False).max()}')

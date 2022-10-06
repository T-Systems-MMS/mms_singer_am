import numpy as np
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils
import audio
from diff.shallow_diffusion import GaussianDiffusion
from diff_dataset import DiffDataset, collate_fn_pad
import hparams as hp

def main(rank=0, world_size=None, restore_checkpoint=None):
  print(f"Initializing rank {rank}...")
  torch.set_num_threads(min(torch.get_num_threads(), 16)) # Max 16 threads
  if world_size:
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

  device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

  train_ds = DiffDataset(hp.mel_ground_truth, hp.mel_fastspeech)
  test_ds = DiffDataset(hp.mel_ground_truth_test, hp.mel_fastspeech_test)

  train_dl = torch.utils.data.DataLoader(
    train_ds,
    batch_size=hp.diff_batch_size,
    collate_fn=collate_fn_pad,
    pin_memory=True,
    num_workers=1,
    sampler=utils.DistributedWeightedSampler(train_ds.weights) if world_size else torch.utils.data.WeightedRandomSampler(train_ds.weights, sum(train_ds.weights))
    #sampler=torch.utils.data.DistributedSampler(train_ds, shuffle=True) if world_size else torch.utils.data.RandomSampler(train_ds)
  )
  test_dl = torch.utils.data.DataLoader(
    test_ds,
    batch_size=hp.diff_batch_size,
    collate_fn=collate_fn_pad,
    pin_memory=True,
    num_workers=0,
    shuffle=False,
    drop_last=False,
  )

  model = GaussianDiffusion(
    denoise_fn = hp.denoise_fn,
    timesteps = hp.diff_timesteps,
    K_step = hp.diff_K_step,
    loss_type = hp.diff_loss_type,
    spec_min = train_ds.spec_min(),
    spec_max = train_ds.spec_max())

  optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=hp.diff_learning_rate,
    weight_decay=hp.diff_weight_decay,
  )

  if restore_checkpoint:
    checkpoint = torch.load(restore_checkpoint, map_location='cpu')
    checkpoint['model']['denoise_fn.position_enc.weight'] = model.denoise_fn.position_enc.weight
    model.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint
    #optimizer.load_state_dict(checkpoint['optimizer'])

  model = model.to(device)
  if world_size:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
  
  #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, hp.diff_max_learning_rate, epochs=hp.diff_epochs, steps_per_epoch=len(train_dl))
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=hp.diff_schedule_step_size*len(train_dl), gamma=hp.diff_schedule_gamma)

  if rank==0:
    logger = SummaryWriter(log_dir=os.path.join('logger', hp.run_name))
    os.makedirs(os.path.join(hp.checkpoint_path, hp.run_name), exist_ok=True)

    print(f'Starting a diffusion training with {len(train_ds)} train and {len(test_ds)} test mels')
    print(f'Number of parameters: {utils.get_param_num(model)} of which trainable {utils.get_param_num(model, True)}')

  for epoch in tqdm(range(hp.diff_epochs)):
    model = model.train()
    for i, db in enumerate(train_dl):
      current_step = i + epoch * len(train_dl)

      gt_mel = db['gt_mel'].to(device, non_blocking=True).repeat(hp.diff_batch_expand, 1, 1)
      fs_dec_input = db['fs_dec_input'].to(device, non_blocking=True).repeat(hp.diff_batch_expand, 1, 1)
      mel_pos = db['mel_pos'].to(device, non_blocking=True).repeat(hp.diff_batch_expand, 1)

      optimizer.zero_grad()
      ret = model(fs_dec_input, mel_targets=gt_mel, mel_pos=mel_pos, infer=False)      
      ret['diff_loss'].backward()
      optimizer.step()  
      scheduler.step()

      if current_step % hp.log_step == 0 and rank==0:
        logger.add_scalar('diff_train/lr', scheduler.get_last_lr()[0], current_step)
        logger.add_scalar('diff_train/loss', ret['diff_loss'].item(), current_step)
        logger.add_scalar('diff_train/epoch', epoch, current_step)

      if epoch % hp.diff_test_step == 0 and i == len(train_dl)-1 and rank==0:
        with torch.no_grad():
          rand_idx = np.random.choice(len(train_ds), 1).item()
          gt_mel = train_ds[rand_idx]['gt_mel']
          fs_dec_input = train_ds[rand_idx]['fs_dec_input'].to(device).unsqueeze(0)
          mel_pos = train_ds[rand_idx]['mel_pos'].to(device).unsqueeze(0)
          fs_mel = train_ds[rand_idx]['fs_mel'].to(device).unsqueeze(0)

          ret = model(fs_dec_input, fs_mel, mel_pos=mel_pos, infer=True)
          mel = ret['mel_out'].cpu()[0]
      
          logger.add_image('diff_train/gt', utils.spec_to_img(gt_mel), epoch, dataformats='HWC')
          logger.add_image('diff_train/fs', utils.spec_to_img(fs_mel.cpu()[0]), epoch, dataformats='HWC')
          logger.add_image('diff_train/pred', utils.spec_to_img(mel), epoch, dataformats='HWC')
          logger.add_audio('diff_train/pred_gl', torch.tensor(audio.mel_to_audio(mel, denorm=True)).unsqueeze(0), epoch, sample_rate = hp.sampling_rate)

      if current_step % hp.save_step == 0 and current_step > 1 and rank==0:
        torch.save({
          'model': model.state_dict() if not world_size else model.module.state_dict(),
          'optimizer': optimizer.state_dict(),
          'scheduler': scheduler.state_dict(),
          'current_step': current_step,
          'current_epoch': epoch
        }, os.path.join(hp.checkpoint_path, hp.run_name, f'checkpoint_{current_step}_diffusion.pth.tar'))
        tqdm.write(f'Saving model at step {current_step}')


    if epoch % hp.diff_test_step == 0 and rank==0:
      model = model.eval()
      with torch.no_grad():
        losses = []
        for i, db in enumerate(test_dl):
          gt_mel = db['gt_mel'].to(device)
          fs_dec_input = db['fs_dec_input'].to(device)
          mel_pos = db['mel_pos'].to(device)

          ret = model(fs_dec_input, mel_targets=gt_mel, mel_pos=mel_pos, infer=False)
          losses.append(ret['diff_loss'].item() * gt_mel.shape[0])
        total_loss = sum(losses)/len(test_ds)
        logger.add_scalar('diff_test/loss', total_loss, epoch)
        tqdm.write(f'Epoch {epoch}, test loss {total_loss}')

        rand_idx = np.random.choice(len(test_ds), 1).item()
        gt_mel = test_ds[rand_idx]['gt_mel']
        fs_mel = test_ds[rand_idx]['fs_mel'].unsqueeze(0).to(device)
        fs_dec_input = test_ds[rand_idx]['fs_dec_input'].unsqueeze(0).to(device)
        mel_pos = test_ds[rand_idx]['mel_pos'].unsqueeze(0).to(device)
        ret = model(fs_dec_input, fs_mel, mel_pos=mel_pos, infer=True)
        mel = ret['mel_out'].cpu()[0]

        logger.add_image('diff_eval/gt', utils.spec_to_img(gt_mel), epoch, dataformats='HWC')
        logger.add_image('diff_eval/fs', utils.spec_to_img(fs_mel.cpu()[0]), epoch, dataformats='HWC')
        logger.add_image('diff_eval/pred', utils.spec_to_img(mel), epoch, dataformats='HWC')
        logger.add_audio('diff_eval/pred_gl', torch.tensor(audio.mel_to_audio(mel, denorm=True)).unsqueeze(0), epoch, sample_rate = hp.sampling_rate)

  
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--restore', type=str, default=None, help='Checkpoint to restore')
  args = parser.parse_args()

  #main(rank=0, world_size=None, restore_checkpoint=args.restore)
  assert torch.cuda.device_count() >= hp.gpus, "More processes than GPUs chosen"
  print(f'Starting a diffusion training with {hp.gpus} workers')
  mp.spawn(main, nprocs=hp.gpus, args=(hp.gpus, args.restore))

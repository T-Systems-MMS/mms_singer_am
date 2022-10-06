import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from multiprocessing import cpu_count
import numpy as np
import argparse
import os
import time
import math

from model import FastSpeech
from loss import DNNLoss
from dataset import BufferDataset, DataLoader
from dataset import get_data_to_buffer, collate_fn_pad
from optimizer import ScheduledOptim
import hparams as hp
import utils
import audio

def main(rank=0, args=None):
    print(f"Hello from rank {rank}")
    world_size = args.num_processes
    os.environ['MASTER_ADDR'] = 'localhost'
    try:
        port = int(hp.run_name) + 12345
    except:
        port = 12355
    os.environ['MASTER_PORT'] = str(port)
    
    # initialize the process group
    torch.set_num_threads(min(torch.get_num_threads(), 16)) # Max 16 threads
    if world_size > 1:
      dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Get device
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    # Define model
    note_conversion = torch.from_numpy(audio.load_note_conversion_table(os.path.join('data', 'note_conversion.csv')))
    model = FastSpeech()
    model = model.to(device)
    model.set_note_conversion(note_conversion)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True).to(device)
    else:
        model = model.to(device)

    # Get buffer
    train_ds = BufferDataset(get_data_to_buffer(hp.label_path, hp.mel_ground_truth, run_asserts=True))
    test_ds = BufferDataset(get_data_to_buffer(hp.label_path_test, hp.mel_ground_truth_test, run_asserts=True))


    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=hp.learning_rate,
                                 betas=hp.betas,
                                 eps=1e-9,
                                 weight_decay=hp.weight_decay)
    scheduled_optim = ScheduledOptim(optimizer,
                                     hp.learning_rate,
                                     hp.n_warm_up_step,
                                     0)
    fastspeech_loss = DNNLoss().to(device)

    # Load checkpoint if exists
    
    if args.restore:
        checkpoint = torch.load(args.restore, map_location='cpu')
        # Copy positional embeddings from current model to allow loading different sizes of positional enc in finetuning
        checkpoint['model']['module.encoder.position_enc.weight'] = model.module.encoder.position_enc.weight
        checkpoint['model']['module.encoder.syllable_pos_enc.weight'] = model.module.encoder.syllable_pos_enc.weight
        checkpoint['model']['module.decoder.position_enc.weight'] = model.module.decoder.position_enc.weight

        model.load_state_dict(checkpoint['model'], strict=True)
        model = model.to(device)
        #optimizer.load_state_dict(checkpoint['optimizer'])
        restore_step = checkpoint.get('current_step', 0)
        restore_epoch = checkpoint.get('current_epoch', 0)
        if not hp.restore_checkpoint_step:
            restore_step = 0
            restore_epoch = 0
        scheduled_optim.n_current_steps = restore_step
        del checkpoint
        print("\n---Model Restored from %s at Step %d Epoch %d---\n" % (args.restore, restore_step, restore_epoch))
    else:
        restore_step = 0
        restore_epoch = 0
        print(f"\n---Start New Training on rank {rank}---\n")
        if not os.path.exists(hp.checkpoint_path):
            os.mkdir(hp.checkpoint_path)


    # Get Training Loader
    if world_size > 1:
        sampler = utils.DistributedWeightedSampler(weights=torch.tensor(train_ds.get_weights()))
    else:
        sampler = torch.utils.data.WeightedRandomSampler(weights=torch.tensor(train_ds.get_weights()), num_samples=round(sum(train_ds.get_weights())))
    sampler = torch.utils.data.BatchSampler(sampler=sampler, batch_size=hp.batch_size, drop_last=False)

    training_loader = DataLoader(train_ds,
                                 sampler=sampler,
                                 num_workers=0,
                                 pin_memory=True)
    total_step = hp.epochs * len(training_loader)

    test_loader = DataLoader(test_ds, 
                             batch_size=hp.batch_size,
                             sampler=torch.utils.data.DistributedSampler(test_ds) if world_size > 1 else torch.utils.data.SequentialSampler(test_ds),
                             collate_fn=collate_fn_pad,
                             drop_last=False)
    total_step_test = (hp.epochs // hp.test_step) * len(test_loader)
    
    logger = None
    if rank==0:
        print('Starting training')
        num_param = utils.get_param_num(model)
        print(f'Number of TTS Parameters: {utils.get_param_num(model)} of which trainable {utils.get_param_num(model, True)}')
        print(f'Test ds: {len(test_ds)}, train ds: {len(train_ds)}')
        print(f'Test dl: {len(test_loader)}, train dl: {len(training_loader)}')

        # Init logger
        if not os.path.exists(hp.logger_path):
            os.mkdir(hp.logger_path)
        os.makedirs(os.path.join(hp.checkpoint_path, hp.run_name), exist_ok=True)

        # Prepare logger
        logger = SummaryWriter(log_dir=os.path.join('logger', hp.run_name))

    # Define Some Information
    Time = np.array([])
    Start = time.perf_counter()

    for epoch in range(restore_epoch, hp.epochs):
        
        # Training
        model = model.train()
    
        start_time = time.perf_counter()
        for i, db in enumerate(training_loader):


            current_step = i + epoch * len(training_loader) + 1

            # Init
            scheduled_optim.zero_grad()

            # Get Data
            character = db["text"].long().to(device, non_blocking=True).squeeze(0)
            note = db["note"].long().to(device, non_blocking=True).squeeze(0)
            src_pos = db["src_pos"].long().to(device, non_blocking=True).squeeze(0)
            duration = db["duration"].int().to(device, non_blocking=True).squeeze(0)
            syllable_duration = db["syllable_duration"].int().to(device, non_blocking=True).squeeze(0)
            syllable_pos = db["syllable_pos"].int().to(device, non_blocking=True).squeeze(0)
            
            mel_pos = db["mel_pos"].long().to(device, non_blocking=True).squeeze(0)
            voiced_target = db["voiced_target"].bool().to(device, non_blocking=True).squeeze(0)
            f0_target = db["f0_target"].float().to(device, non_blocking=True).squeeze(0)
            mel_target = db["mel_target"].float().to(device, non_blocking=True).squeeze(0)
            max_mel_len = db["mel_max_len"][0]

            # Forward
            mel_output, mel_postnet_output, _, f0_output, voiced_output, duration_predictor_output, _ = model(character,
                                                                              note,
                                                                              syllable_duration,
                                                                              syllable_pos,
                                                                              src_pos,
                                                                              mel_pos=mel_pos,
                                                                              mel_max_length=max_mel_len,
                                                                              length_target=duration,
                                                                              voiced_target=voiced_target,
                                                                              stl_target=mel_target if hp.use_gst else None)
            

            # Cal Loss
            mel_loss, mel_postnet_loss, duration_loss, f0_loss, voiced_loss = fastspeech_loss(mel_output,
                                                                        mel_postnet_output,
                                                                        duration_predictor_output,
                                                                        f0_output,
                                                                        voiced_output,
                                                                        mel_target,
                                                                        duration,
                                                                        f0_target,
                                                                        voiced_target)
            total_loss = mel_loss + mel_postnet_loss + duration_loss + f0_loss + voiced_loss

            assert not torch.isnan(total_loss), f'NaN loss, cancelling training'

            # Backward
            total_loss.backward()

            # Clipping gradients to avoid gradient explosion
            nn.utils.clip_grad_norm_(
                model.parameters(), hp.grad_clip_thresh)

            # Update weights
            if args.frozen_learning_rate > 0:
                scheduled_optim.step_and_update_lr_frozen(
                    args.frozen_learning_rate)
            else:
                scheduled_optim.step_and_update_lr()

            if epoch % hp.test_step == 0 and i == 0 and rank==0:
                batch_idx = np.random.randint(0, mel_output.shape[0])
                mel = mel_postnet_output[batch_idx].detach().cpu()
                logger.add_image('train/mel', utils.spec_to_img(mel_output[batch_idx].detach().cpu(), f0_output[batch_idx].detach().cpu(), voiced_output[batch_idx].detach().cpu()), epoch, dataformats='HWC')
                logger.add_image('train/mel_postnet', utils.spec_to_img(mel), epoch, dataformats='HWC')
                logger.add_image('train/mel_target', utils.spec_to_img(mel_target[batch_idx].cpu(), f0_target[batch_idx].cpu(), voiced_target[batch_idx].cpu()), epoch, dataformats='HWC')
                logger.add_audio('train/mel_postnet_gl', torch.tensor(audio.mel_to_audio(mel, denorm=True)).unsqueeze(0), epoch, sample_rate=hp.sampling_rate)
                logger.add_audio('train/mel_target_gl', torch.tensor(audio.mel_to_audio(mel_target[batch_idx].cpu(), denorm=True)).unsqueeze(0), epoch, sample_rate=hp.sampling_rate)

            # Print
            if current_step % hp.log_step == 0 and rank==0:
                # Logger
                t_l = total_loss.item()
                m_l = mel_loss.item()
                m_p_l = mel_postnet_loss.item()
                d_l = duration_loss.item()
                f_l = f0_loss.item()
                v_l = voiced_loss.item()

                Now = time.perf_counter()
                logger.add_scalar('train/total_loss', t_l, current_step)
                logger.add_scalar('train/mel_loss', m_l, current_step)
                logger.add_scalar('train/mel_postnet_loss', m_p_l, current_step)
                logger.add_scalar('train/duration_loss', d_l, current_step)
                logger.add_scalar('train/f0_loss', f_l, current_step)
                logger.add_scalar('train/voiced_loss', v_l, current_step)
                logger.add_scalar('train/framerate', hp.gpus/((Now - start_time)/mel_output.shape[0]), current_step)
                logger.add_scalar('train/lr', scheduled_optim.get_learning_rate(), current_step)
                logger.add_scalar('epoch', epoch, current_step)
                start_time = Now
                
                str1 = "Epoch [{}/{}], Step [{}/{}]:".format(
                    epoch+1, hp.epochs, current_step, total_step)
                str2 = "Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Duration Loss: {:.4f}, F0 Loss: {:.4f}, V/uV Loss: {:.4f};".format(
                    m_l, m_p_l, d_l, f_l, v_l)
                str3 = "Current Learning Rate is {:.9f}.".format(
                    scheduled_optim.get_learning_rate())
                str4 = "Time Used: {:.3f}s, Estimated Time Remaining: {:.3f}s.".format(
                    (Now-Start), (total_step-current_step)*np.mean(Time))

                print("\n" + str1)
                print(str2)
                print(str3)
                print(str4)

            if (current_step % hp.save_step == 0 or current_step == total_step) and rank==0:
                torch.save({
                    'model': model.state_dict(), 
                    'optimizer': optimizer.state_dict(),
                    'current_step': current_step,
                    'current_epoch': epoch
                }, os.path.join(hp.checkpoint_path, hp.run_name, 'checkpoint_%d.pth.tar' % current_step))
                print("save model at step %d ..." % current_step)


            end_time = time.perf_counter()
            Time = np.append(Time, end_time - start_time)
            if len(Time) == hp.clear_Time:
                temp_value = np.mean(Time)
                Time = np.delete(
                    Time, [i for i in range(len(Time))], axis=None)
                Time = np.append(Time, temp_value)
        #if True:
        if epoch % hp.test_step == 0:
            t_l = []
            m_l = []
            m_p_l = []
            d_l = []
            d_d = []
            f_l = []
            v_l = []
            v_d = []

            # Eval mode
            model = model.eval()
            current_test_step = 0
            for i, db in enumerate(test_loader):
                start_time = time.perf_counter()


                # Get Data
                character = db["text"].long().to(device)
                note = db["note"].long().to(device)
                mel_target = db["mel_target"].float().to(device)
                f0_target = db['f0_target'].float().to(device)
                voiced_target = db['voiced_target'].bool().to(device)
                duration = db["duration"].int().to(device)
                syllable_duration = db["syllable_duration"].int().to(device)
                syllable_duration_raw = db["syllable_duration_raw"].int().to(device)
                syllable_pos = db["syllable_pos"].int().to(device)                    
                mel_pos = db["mel_pos"].long().to(device)
                src_pos = db["src_pos"].long().to(device)
                max_mel_len = db["mel_max_len"]

                
                
                with torch.no_grad():

                    # Forward
                    mel_output, mel_postnet_output, mel_pos_output, f0_output, voiced_output, duration_predictor_output, _ = model(character,
                                                                                      note,
                                                                                      syllable_duration,
                                                                                      syllable_pos,
                                                                                      src_pos,
                                                                                      mel_pos=mel_pos,
                                                                                      mel_max_length=max_mel_len,
                                                                                      length_target=duration,
                                                                                      voiced_target=voiced_target,
                                                                                      stl_target=mel_target if hp.use_gst else None)

                    voiced_diff = (voiced_output.detach().sigmoid().round().sum(axis=1) - voiced_target.sum(axis=1)).mean()
                    

                    # Cal Loss
                    mel_loss, mel_postnet_loss, duration_loss, f0_loss, voiced_loss = fastspeech_loss(mel_output,
                                                                                mel_postnet_output,
                                                                                duration_predictor_output,
                                                                                f0_output,
                                                                                voiced_output,
                                                                                mel_target,
                                                                                duration,
                                                                                f0_target,
                                                                                voiced_target)

                    if world_size > 1:
                        dist.all_reduce(mel_loss, dist.ReduceOp.SUM)
                        mel_loss /= world_size
                        dist.all_reduce(mel_postnet_loss, dist.ReduceOp.SUM)
                        mel_postnet_loss /= world_size
                        dist.all_reduce(duration_loss, dist.ReduceOp.SUM)
                        duration_loss /= world_size
                        dist.all_reduce(f0_loss, dist.ReduceOp.SUM)
                        f0_loss /= world_size
                        dist.all_reduce(voiced_loss, dist.ReduceOp.SUM)
                        voiced_loss /= world_size

                    total_loss = mel_loss + mel_postnet_loss + duration_loss + f0_loss + voiced_loss

                # Logger
                if rank==0:
                    t_l.append(total_loss.item())
                    m_l.append(mel_loss.item())
                    m_p_l.append(mel_postnet_loss.item())
                    d_l.append(duration_loss.item())

                    f_l.append(f0_loss.item())
                    v_l.append(voiced_loss.item())
                    v_d.append(voiced_diff.item())


                if i==0 and rank==0:
                    
                    # Forward again but unguided for image generation
                    batch_idx = np.random.choice(character.shape[0], 1, replace=False)
                    if hp.use_gst:
                        batch_idx = batch_idx.repeat(hp.gst_token_num + 1)
                        gst_weights = torch.diag_embed(torch.ones(hp.gst_token_num, device=device, dtype=torch.float32))
                        gst_weights = torch.cat((torch.zeros(1,hp.gst_token_num, device=device, dtype=torch.float32), gst_weights))

                    with torch.no_grad():
                        mel_output, mel_postnet_output, mel_pos_output, f0_output, voiced_output, _, _ = model(character[batch_idx],
                                                                                          note[batch_idx],
                                                                                          syllable_duration[batch_idx],
                                                                                          syllable_pos[batch_idx],
                                                                                          src_pos[batch_idx],
                                                                                          stl_weights=gst_weights if hp.use_gst else None,
                                                                                          syllable_dur_guidance=syllable_duration_raw[batch_idx])
                    
                    mel_target = mel_target[batch_idx]
                    voiced_target = voiced_target[batch_idx]
                    f0_target = f0_target[batch_idx]
                    mel = mel_postnet_output[0].detach().cpu()
                    logger.add_image('eval/mel', utils.spec_to_img(mel_output[0].detach().cpu(), f0_output[0].detach().cpu(), voiced_output[0].detach().cpu()), epoch, dataformats='HWC')
                    logger.add_image('eval/mel_postnet', utils.spec_to_img(mel), epoch, dataformats='HWC')
                    logger.add_image('eval/mel_target', utils.spec_to_img(mel_target[0].cpu(), f0_target[0].cpu(), voiced_target[0].cpu()), epoch, dataformats='HWC')
                    logger.add_audio('eval/mel_postnet_gl', torch.tensor(audio.mel_to_audio(mel, denorm=True)).unsqueeze(0), epoch, sample_rate=hp.sampling_rate)
                    logger.add_audio('eval/mel_target_gl', torch.tensor(audio.mel_to_audio(mel_target[0].cpu(), denorm=True)).unsqueeze(0), epoch, sample_rate=hp.sampling_rate)
                    if hp.use_gst:
                        for i in range(hp.gst_token_num):
                            logger.add_audio(f'eval/mel_gst_{i}', torch.tensor(audio.mel_to_audio(mel_postnet_output[i+1].detach().cpu(), denorm=True)).unsqueeze(0), epoch, sample_rate=hp.sampling_rate)

                    with torch.no_grad():
                        mel_output, mel_postnet_output, mel_pos_output, f0_output, voiced_output, _, _ = model(character[batch_idx],
                                                                                          note[batch_idx],
                                                                                          syllable_duration[batch_idx],
                                                                                          syllable_pos[batch_idx],
                                                                                          src_pos[batch_idx])
                    d_d.append(mel_output.shape[1]-mel_target.shape[1])
                    mel = mel_postnet_output[0].cpu()
                    logger.add_image('eval/unguided_mel_postnet', utils.spec_to_img(mel, f0_output[0].cpu(), voiced_output[0].cpu()), epoch, dataformats="HWC")
                    logger.add_audio('eval/unguided_mel_postnet_gl', torch.tensor(audio.mel_to_audio(mel, denorm=True)).unsqueeze(0), epoch, sample_rate=hp.sampling_rate)
        
            if rank==0:
                t_l = np.mean(t_l)
                m_l = np.mean(m_l)
                m_p_l = np.mean(m_p_l)
                d_l = np.mean(d_l)
                d_d = np.mean(d_d)
                f_l = np.mean(f_l)
                v_l = np.mean(v_l)
                v_d = np.mean(v_d)

                logger.add_scalar('eval/total_loss', t_l, epoch)
                logger.add_scalar('eval/mel_loss', m_l, epoch)
                logger.add_scalar('eval/mel_postnet_loss', m_p_l, epoch)
                logger.add_scalar('eval/duration_loss', d_l, epoch)
                logger.add_scalar('eval/f0_loss', f_l, epoch)
                logger.add_scalar('eval/voiced_loss', v_l, epoch)
                logger.add_scalar('eval/voiced_diff', v_d, epoch)
                logger.add_scalar('eval/duration_diff', d_d, epoch)
                print("Epoch {:d}--- Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Duration Loss: {:.4f}, Duration diff {:.4f}, F0 Loss {:.4f}, V/UV Loss {:.4f};".format(
                    epoch, m_l, m_p_l, d_l, d_d, f_l, v_l))
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore', type=str, default=None)
    parser.add_argument('--frozen_learning_rate', type=float, default=-1)
    parser.add_argument('--num_processes', type=int, default=hp.gpus)
    args = parser.parse_args()
    
    if args.num_processes == 1:
        main(0, args)
    else:
        assert torch.cuda.device_count() >= args.num_processes, "More processes than GPUs chosen"
        print(f'Starting a hifisinger training with {args.num_processes} workers')
        mp.spawn(main, nprocs=args.num_processes, args=(args,))

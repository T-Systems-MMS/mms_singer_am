import eval
import hparams as hp
import re
import os
import shutil
import argparse 

class Args():
    def __init__(self, train=True, checkpoint = None):
        self.data_path = hp.label_path if train else hp.label_path_test
        self.textnotegen = False
        self.split_textnotegen = 'none'
        self.guidance = 'phoneme'

        if checkpoint is None:
            # try opening the best
            try:
                with open(os.path.join(hp.checkpoint_path, hp.run_name, 'best'), 'r') as f:
                    checkpoint = os.path.join(hp.checkpoint_path, hp.run_name, f.read().strip())
            except:
                # Find the latest checkpoint
                files = os.listdir(os.path.join(hp.checkpoint_path, hp.run_name))
                files = [(f, re.search(r'checkpoint_(\d+)\.pth\.tar', f)) for f in files]
                files = [(f, int(m.group(1))) for f,m in files if m is not None]
                files.sort(key=lambda tup: tup[1])
                checkpoint = os.path.join(hp.checkpoint_path, hp.run_name, files[-1][0])

        self.checkpoint_path = checkpoint
        self.alpha = 1.0
        self.result_path = hp.mel_fastspeech if train else hp.mel_fastspeech_test
        self.use_gl = False
        self.style_token_weights = None
        self.style_token_gt = True
        self.style_token_target = None
        self.save_norm_mel = True
        self.choir_mode = None
        self.choir_mode_variance = 0
        self.diffusion_decoder = None

def main(checkpoint=None):
    test_args = Args(train=False, checkpoint=checkpoint)
    train_args = Args(train=True, checkpoint=checkpoint)

    print('Creating test ds')
    print(test_args.__dict__)
    eval.main(test_args)
    [os.remove(os.path.join(test_args.result_path, f)) for f in os.listdir(test_args.result_path) if not f.endswith('.pkl')]

    
    print('Creating train ds')
    print(train_args.__dict__)
    eval.main(train_args)
    [os.remove(os.path.join(train_args.result_path, f)) for f in os.listdir(train_args.result_path) if not f.endswith('.pkl')]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process a dataset so a diffusion model can be trained on it')
    parser.add_argument('--checkpoint', type=str, default=None, help='The fastspeech model to load, defaults to the latest model')
    args = parser.parse_args()
    main(args.checkpoint)

import eval
import hparams as hp
import re
import os
import shutil
import argparse 

class Args():
    def __init__(self, result_path, train=True, checkpoint = None, checkpoint_diff = None, use_diff=False):
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
        self.result_path = result_path
        self.use_gl = False
        self.style_token_weights = None
        self.style_token_gt = True
        self.style_token_target = None
        self.save_norm_mel = True
        self.choir_mode = None
        self.choir_mode_variance = 0

        if checkpoint_diff is None:
            # try opening the best
            try:
                with open(os.path.join(hp.checkpoint_path, hp.run_name, 'best_diff'), 'r') as f:
                    checkpoint_diff = os.path.join(hp.checkpoint_path, hp.run_name, f.read().strip())
            except:
                # Find the latest checkpoint
                files = os.listdir(os.path.join(hp.checkpoint_path, hp.run_name))
                files = [(f, re.search(r'checkpoint_(\d+)_diffusion\.pth\.tar', f)) for f in files]
                files = [(f, int(m.group(1))) for f,m in files if m is not None]
                files.sort(key=lambda tup: tup[1])
                checkpoint_diff = os.path.join(hp.checkpoint_path, hp.run_name, files[-1][0])

        self.diffusion_decoder = checkpoint_diff if use_diff else None

def main(args):
    test_args = Args(args.result_path_test, train=False, checkpoint=args.checkpoint, checkpoint_diff=args.checkpoint_diff, use_diff=bool(args.use_diff))
    train_args = Args(args.result_path_train, train=True, checkpoint=args.checkpoint, checkpoint_diff=args.checkpoint_diff, use_diff=bool(args.use_diff))

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
    parser.add_argument('--use_diff', type=bool, default=False, help='Whether to use the diffusion decoder')
    parser.add_argument('--checkpoint_diff', type=str, default=None, help='The diffusion model to use')
    parser.add_argument('--result_path_train', type=str, default=os.path.join(hp.dataset, 'voc_mel'), help='Where to save the result files for train data')
    parser.add_argument('--result_path_test', type=str, default=os.path.join(hp.dataset_test, 'voc_mel'), help='Where to save the result files for test data')
    args = parser.parse_args()
    main(args)

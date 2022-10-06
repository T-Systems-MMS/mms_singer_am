import shutil
import os
from data import k3
from data import csd
import hparams as hp


def preprocess_csd(filename):
    print('Starting preprocessing')
    in_dir = filename
    target_dir = hp.dataset_processed
    
    for d in ['csv', 'wav_mono', 'snippets', 'textgrid', 'snippets_test']:
        if os.path.exists(os.path.join(target_dir, d)):
            shutil.rmtree(os.path.join(target_dir, d))

    if hp.dataset_format == "K3":
        k3.build_from_path(in_dir, target_dir)
    elif hp.dataset_format == "CSD":
        csd.build_from_path(in_dir, target_dir)
    else:
        raise "Invalid dataset format"

def main():
    preprocess_csd(hp.dataset_raw)


if __name__ == "__main__":
    main()

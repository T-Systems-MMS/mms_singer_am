# MMS-Singer Implementation

This implementation has been used to generate the AI Voice for the opera [Chasing Waterfalls](https://www.semperoper.de/spielplan/stuecke/stid/chasing-waterfalls/62127.html), which was staged in Semperoper Dresden in September 2022. At [T-Systems MMS](https://blog.t-systems-mms.com/digital-stories/ki-goes-semperoper-dresden-so-spielt-die-musik-der-zukunft), we built and trained a singing voice synthesis system. For the opera, we used an in-house dataset, which we can not provide due to licensing issues. As a result, we will also not provide any pre-trained models, but we rebuilt our preprocessing pipeline to also support the [CSD dataset](https://zenodo.org/record/4785016), a freely available singing dataset to enable experimentation with our implementation. 

From an architecture perspective, this implementation is loosely based on [hifisinger](https://arxiv.org/abs/2009.01776) and [diffsinger](https://arxiv.org/abs/2105.02446). We forked our implementation from xzmyz's [FastSpeech implementation](https://github.com/xcmyz/FastSpeech). Furthermore we adapted the GST implementation from [KinglittleQ](https://github.com/KinglittleQ/GST-Tacotron) to work with FastSpeech. The diffusion decoder is adapted from the DiffSinger implementation by [MoonInTheRiver](https://github.com/MoonInTheRiver/DiffSinger). 

The code is published in three repositories, you are currently looking at the acoustic model repository but also have a look at the [frontend](https://github.com/T-Systems-MMS/mms_singer_frontend) and [vocoder](https://github.com/T-Systems-MMS/mms_singer_vocoder)

## Adapting GSTs to work with Fastspeech

We adapted the [Style Token Idea](https://arxiv.org/abs/1803.09017) as an unsupervised approach to add style controllability. While FastSpeech2 offers some controllability through the Variance Adaptor, it only offers specific predefined styles. The styles found through the unsupervised approach turned out more helpful for our artist partners, as they tend to encode information such as vibrato, clarity, anger, ... In our experiments, we found 4 style tokens to be a good number, with a bigger dataset it might be possible to train on more tokens.

## Switching to local attention

Global attention is not a sensible attention for a synthesis system which is trained in snippets, because the result is actually a snippet-local attention. Instead, we train with the local attention concept known from [Longformers](https://arxiv.org/abs/2004.05150), which gives us more explicit control over the attention window without having to respect hardware constraints. We use the implementation by [lucidrains](https://github.com/lucidrains/local-attention). We also experimented with multiple linear attention variants implemented by [idiap](https://github.com/idiap/fast-transformers/), but found that to severely degrade quality.

## Advanced preprocessing

We found preprocessing to be the most important contributor to final model quality, hence we implement a sophisticated preprocessing pipeline. Most notably, the literature does not distinguish between time-aligned midi and midi-converted sheet music. Our preprocessing tries to prepare the time-aligned data used for training to be as compatible as possible to midi-converted sheet music, which is the actual input the model will see during production.

## Inference-time options

To satisfy the artistic requirements, we added inference time functionalities. Most notably a choir mode, which synthesizes multiple voices with slightly randomized style tokens, making it sound like a choir is singing and syllable guidance, which accumulates duration predictor errors across a syllable and applies a correction term to exactly match the syllable length from the midi. The latter is required if the model should sing aligned with other instruments, as it must sing each note on time.

# How to use this repository

You can use this repository to train your own SVS, but you will have to resort to a publicly available dataset. We can not provide the dataset nor the model used in the opera, but added routines to work on the [Childrens Song Dataset](https://zenodo.org/record/4785016)

## Preprocess

* Download the CSD dataset (or any other dataset in the same format) to data/CSD. Note that we used our own dataset for the opera.
* Run `cd data && bash csd_preprocess.sh` (note that conda is required for this). This will run the montreal forced aligner to get phoneme alignment information.
* Run `python preprocess.py`. This will do multiple things, depending on the dataset format. As a result, this is quite a lengthy process. You can activate preprocessing parallelism in hparams.py but be prepared to wait multiple hours. If you use CSD format, it will
    * Convert the labels from the CSD dataset into a packed format, where there is a `<nosound>` between every note and with phoneme-level annotations instead of the syllable level annotations from CSD
    * Find the optimal split into snippets that maximizes snippet length while trying to split only in `<nosound>` locations 
    * Split long notes into multiple consecutive ones with the same pitch and phoneme. In our setting we knew notes would never exceed a length so longer notes were a burden on the duration predictor
    * Creating a test-train split
    * Performing data augmentation. Depending on the settings in hparams.py it is possible to use pitch shift and time stretch effects. As these effects degrade audio quality, you can specify a sampling probability for augmented data.
    * Processing the waveform into mel spectrograms and extracting f0 information
* Check upon the result by running `python dataset.py` to get some dataset statistics
* Files are in `CSD_processed/snippets`


## Train

* Adjust hyperparameters in hparams.py
* Run `python train.py` to train the transformer acoustic model
* Check upon the training progress with tensorboard --logdir=logger and choose a model checkpoint
* When you have reached a good model, call `python make_diff_dataset.py` with the model checkpoint you want to use to prepare a dataset for diffusion training. If no model was specified, the script will use the last checkpoint, which is not necessarily the best. Make sure to train the diffusion decoder with the same model you plan to use for inference.
* Run `python train_diff.py` to train the diffusion decoder
* Check upon the training progress with tensorboard and choose a model checkpoint
* Run `python make_vocoder_dataset.py` to create a vocoder dataset. You can create a dataset straight from the transformer acoustic model or from the diffusion decoder, specify the models that you plan to use for inference here.
* Follow the instructions in the [vocoder repository](https://github.com/T-Systems-MMS/mms_singer_vocoder) to train the vocoder.

## Troubleshooting

Some common problems during training can be fixed quite easily:
* CUDA out of memory - decrease the batch size or attention context to fit your GPU memory. Note that lower batch-size training is prone to overfit, you can add multiple GPUs to train with to increase effective batch size. Lower attention context might result in lower quality beyond a certain threshold but should not affect quality above that. Also, lower attention contexts might improve generalization
* Dataset assertions when running dataset.py or starting a training - Some snippets might fail dataset assertions during the beginning of the training. As long as this only concerns a few snippets, you can ignore it. These snippets will be excluded from training. If there are many errors, try to investigate where the assertion might be coming from as your dataset size might end up too small for a sensible training.
* Loss explosions - Reduce the learning rate or increase the batch size
* Notes or phonemes out of vocabulary - increase the corresponding vocabulary size in hparams.py

## Eval

The code supports two formats for evaluation. You can either run it on parts of the dataset and use the format of a preprocessed dataset for input, or use a format which we refer to as **textnotegen** format. The textnotegen format consists of a folder with a `melody.mid` which contains the melody that should be sung in midi format and a `txt_punctuation.txt` which contains the phonemes to be sung over that melody.
* Run `python eval.py --help` to get all available options for eval
* Run `python eval.py` with the models used during training

Or check out the [frontend repository](https://github.com/T-Systems-MMS/mms_singer_frontend), which provides a browser frontend for inference.

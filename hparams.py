import os

run_name="097"

# Dataset
#dataset_raw = os.path.join('data', 'LJSpeech-1.1')
#dataset_processed = os.path.join('data', 'LJSpeech_processed')
#dataset = dataset_processed

dataset_raw = os.path.join('data', 'CSD', 'english')
dataset_processed = os.path.join('data', 'CSD_processed')

#dataset_raw = os.path.join('data', 'K3')
#dataset_processed = os.path.join('data', 'K3_processed')

dataset = os.path.join(dataset_processed, 'snippets')
mel_ground_truth = os.path.join(dataset, 'mel')
label_path = os.path.join(dataset, 'csv')
mel_fastspeech = os.path.join(dataset, 'fs_mel')

dataset_test = os.path.join(dataset_processed, 'snippets_test')
mel_ground_truth_test = os.path.join(dataset_test, 'mel')
label_path_test = os.path.join(dataset_test, 'csv')
mel_fastspeech_test = os.path.join(dataset_test, 'fs_mel')

phoneme_dict = os.path.join(os.path.dirname(__file__), 'data', 'ljspeech_unstressed_phoneme_dict.pkl')

# Data augmentations are denoted as (param, sample_propability)
# Params for pitch shift are the semitones up/down
# Params for time stretch is the time stretch factor
# The sample propability
augment_pitch_shift = [(1, 0.1), (-1, 0.1), (2, 0.05), (-2, 0.05), (3, 0.05), (-3, 0.05)]
augment_time_stretch = [(1.05, 0.1), (0.98, 0.1), (1.1, 0.1), (0.95, 0.05), (1.15, 0.05)]

dataset_format = "CSD" # One of CSD|K3 - determines which preprocessing script to use
preprocess_workers = 32 # Number of parallel preprocessing workers, e.g. number of CPUs
max_snippet_length = 17  # Snippets will be split to this length, in seconds
max_note_length = 4 # Will be split up during preprocessing, in seconds
remove_nosounds = 0.9 # Removes 70% of nosounds in csv labels

# Which fraction of the dataset will be used for training
test_train_split_seed = 0
test_train_split = 0.95

# Mel generation parameters
num_mels = 80
sampling_rate = 44100
filter_length = 1024
win_length = 1024 # ca 25 ms
hop_length = 256 # ca 6 ms
center_mels = False
mel_fmin = 70.0
mel_fmax = 10000.0

# Normalization parameters
mel_norm_file = os.path.join(os.path.dirname(__file__), 'data', 'mel_norm.pkl')
pitch_norm_min = 4.3
pitch_norm_max = 6.4
intensity_norm_min = -40.3
intensity_norm_max = 86.5
hnr_norm_min = 0.
hnr_norm_max = 60.7


# FastSpeech
# F0 information in unvoiced sections is unclear, hence do not calculate a gradient in these regions
guide_f0_with_voiced_targets = True
# Phoneme vocabulary size
vocab_size = 72
# Note vocab should correspond to the highest note in any song
note_vocab_size = 80
# Whether to bin note durations. This way, the duration embedding has less entries and is easier to train but some precision
# in the duration embedding is lost. 
bin_durations = True
bin_durations_min = 1
bin_durations_max = 800 # If you use split_long_notes during preprocessing, you can set this to split_long_notes * sampling_rate / hop_length
bin_durations_count = 300
# Note duration vocab, in case of binning it should be duration bins + 1
syllable_duration_vocab_size = 301
if bin_durations:
    assert syllable_duration_vocab_size == bin_durations_count + 1
duration_loss_norm = 0.01 # The duration loss should be of smaller magnitude than the decoder loss

# The maximal sequence length for mel targets
max_seq_len = 4000
# The maximal sequence length for text/note inputs
max_seq_len_txt = 200
phoneme_pos_vocab_size = max_seq_len_txt + 1
syllable_pos_vocab_size = max_seq_len_txt + 1
relative_src_pos = False # Whether src_pos is relative to the syllable or absolute in the snippet
limit_seq_len = True # Whether to limit the sequence length during inference

# Encoder parameters
phoneme_emb_dim = 128 # Embedding size for phoneme-level information (phoneme, phoneme pos)
syllable_emb_dim = 256 # Embedding size for syllable-level information (note, duration, syll pos)
encoder_dim = 384 # Total encoder dimension, should be phoneme_emb_dim + syllable_emb_dim
encoder_n_layer = 6 # How many encoder layers to use (down to 4 worked as well)
encoder_head = 2 # The number of attention heads, or how many positions a layer can attend to. 2 seems like a sweet spot, 1 and 3 both give terrible results
encoder_conv1d_filter_size = 128 # The size of the conv filter bank. In default transformers, this is a FC layer but for fastspeech it is a 1D conv with large filter
encoder_attn_type = 'local' # Whether to use local or global attention. Local scales better and does not bring a performance penalty if the context is large enough
encoder_attn_local_context = 16 # At least 16 tokens in each direction.
encoder_attn_local_pos = False # Whether to add a local positional embedding in each attention calculation

# Decoder parameters
decoder_dim = 384 # This should be equal to the encoder dimension
decoder_n_layer = 6 # How many layers, 6 or 8 works fine
decoder_head = 2 # The number of attention heads seems to only work with 2.
decoder_conv1d_filter_size = 1536 # The decoder filter banks are larger than the encoder banks, as the decoder operates in mel dimension
decoder_attn_type = 'local' # A local attention mechanism brings scalability and generalization advantages
decoder_attn_local_context = 256 # The size of the local context in each direction
decoder_attn_local_pos = False # Whether to add a local positional embedding to each attention step

# For both the encoder and decoder, which conv kernel and padding sizes to use
fft_conv1d_kernel = (9, 1)
fft_conv1d_padding = (4, 0)

# The duration predictor consists of a 1D Conv with the following sizes
duration_predictor_filter_size = 384
duration_predictor_kernel_size = 3

# Dropout is being used in multiple places, this changes all dropouts globally
dropout = 0.1

# GST
use_gst = True
# The reference encoder is a conv net that scales down an arbitrarily sized mel target to find gst_token_num token weights
ref_enc_filters = [32,32,64,64,128,128]
ref_enc_size = (3,3)
ref_enc_stride = (2,2)
ref_enc_padding = (1,1)
ref_enc_gru_size = decoder_dim // 2
# If you have a lot and high quality data, you can train more than just 4 GSTs
gst_token_num = 4
# Setting this to something else than 1 does not make sense with the current inference implementation
gst_head = 1

# Train
gpus = 1  # How many DistributedDataParallel processes to spawn, used by both train.py and train_diff.py
checkpoint_path = "./model_new"
logger_path = "./logger"
epochs = 250 # After how many epochs to stop the training. With intensive data augmentation, this should be lowered but generally you can train until you see overfit
restore_checkpoint_step = False # When loading a checkpoint, whether to start from the last step (resume training) or start with a fresh optimizer from step zero (sensible for finetuning)

#learning_rate = decoder_dim**-0.5
learning_rate = 0.016 # The learning rate
n_warm_up_step = 4000 # Step up the LR in the first few steps
batch_size = 28 # Batch size, use the maximum possible with your GPU
weight_decay = 1e-6 # Adam weight decay
betas = (0.9, 0.98) # Adam betas
grad_clip_thresh = 1.0 # Use gradient clipping if grad norms are larger than this

save_step = 2500 # After how many steps to save a model
log_step = 10 # After how many steps to log to tensorboard
test_step = 5 # After how many epochs (not steps) to perform an evaluation
clear_Time = 20 # Parameter for the time left calculation


# Diffusion model hyperparameters

# Either l1 or l2, the loss for the p_losses
diff_loss_type = 'l1'
# A cosine schedule gives a lower step size in the low-noise regions and is generally a good idea
beta_schedule_type = 'cosine'
# Whether to start inference from gaussian noise or from fastspeech mel (shallow diffusion)
gaussian_start = False 

# FFT uses the same decoder architecture as in the AM decoder as the denoise function
denoise_fn = 'fft' # fft|wavenet

# Parameters are similar to the AM decoder
diff_decoder_n_layer = 6
diff_decoder_head = 2
diff_decoder_conv1d_filter_size = 1536
diff_dropout = 0.1

diff_timesteps = 100
diff_K_step = 70 # The shallow diffusion step



# Learning rate
diff_learning_rate = 1e-4
#diff_max_learning_rate = 1e-2
diff_schedule_gamma = 0.95 # Multiply the LR by this gamma every step_size steps
diff_schedule_step_size = 10
diff_weight_decay = 1e-2 # Adam weight decay
diff_batch_size = 32
# How often to repeat the batch (they will have different noise factors) - you can use that if your CPU is slow and the GPU fast
diff_batch_expand = 1
diff_epochs = 500 # How many epochs to train the diffusion decoder. By experience, overfit happens later than in the AM training
diff_test_step = 5 # in epochs

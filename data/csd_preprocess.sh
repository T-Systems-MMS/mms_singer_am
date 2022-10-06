#!/bin/bash
# Parses waves to mono and extracts f0 information
# To use, install gnu parallel (apt install parallel) and reaper
# For reaper install instructions, see https://github.com/google/REAPER
# Make sure the reaper executable is in the PATH
# Also make sure conda is installed and in the PATH

export CSD_PATH=${CSD_PATH:-./CSD/english}
export CSD_PROCESSED_PATH=${CSD_PROCESSED_PATH:-./CSD_processed}


if [ -d ${CSD_PATH} ]
then
    echo "Loading from ${CSD_PATH}"
else
    echo "${CSD_PATH} not found! Please download the CSD dataset to that location"
    exit -1
fi

echo "Saving into ${CSD_PROCESSED_PATH}"
mkdir -p ${CSD_PROCESSED_PATH}
# stereo to mono conversion now part of make_mfa_corpus.py


wget https://gist.githubusercontent.com/dimitre/439f5ab75a0c2e66c8c63fc9e8f7ea77/raw/65cc3bf37ac4d0b4ab268b41bf60077eae48e25e/note_freq_440_432.csv -O ${CSD_PROCESSED_PATH}/note_conversion.csv

#rm -rf wav_mono
#mkdir wav_mono

#echo "Converting wav files to mono"
#ls wav | parallel -I% ffmpeg -i wav/% -ac 1 wav_mono/%

#rm -rf f0
#mkdir f0

#echo "Parsing f0 information"
# The -e flag should be equal to the mel hop length in ms
#ls wav_mono | cut -d '.' -f 1 | parallel -I% reaper -i wav_mono/%.wav -f f0/%.f0 -a -e 0.004988662131519

echo "Computing alignments"
CONDA_ALWAYS_YES=true conda create -n aligner -c conda-forge montreal-forced-aligner
eval "$(conda shell.bash hook)"
# <<< conda initialize <<<
conda activate aligner
pip install pandas pydub
mfa model download acoustic english_us_arpa
python3 make_mfa_corpus.py --in_dir=${CSD_PATH} --target_dir=${CSD_PROCESSED_PATH}/mfa_csd_corpus
# mfa validate -v mfa_csd_corpus mfa_csd_dict.txt english
# mfa_output folder must not exist, mfa will create the folder
rm -rf ${CSD_PROCESSED_PATH}/mfa_output
mfa align ${CSD_PROCESSED_PATH}/mfa_csd_corpus mfa_csd_dict.txt english_us_arpa ${CSD_PROCESSED_PATH}/mfa_output --clean --beam 400 --retry_beam 1000
conda deactivate

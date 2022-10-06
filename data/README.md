# Updates by Max

The fully processed CSD will be available in CSD_processed.
I do not yet know how to run the Montreal Forced Aligner within a script.
Until I figure that out, here is how to start the preprocessing:

1. Run make_mfa_corpus.py
2. Run Montreal Forced Aligner (see csd_preprocess.sh)
3. Run preprocess.py (outside of this folder)
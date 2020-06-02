#!/bin/bash

CDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

export CUDA_VISIBLE_DEVICES=0
export MPLBACKEND="Agg"

# Constraints only
python $CDIR/main_train_codec.py --with-envelope=1.0 --learning-rate=0.001 --output-name="codec-constraints-only"
python $CDIR/main_analyze.py --input-name="codec-constraints-only"

# Constraints + GM
python $CDIR/main_train_gm.py
python $CDIR/main_train_codec.py --with-envelope --with-gm --output-name="codec-constraints-gm"
python $CDIR/main_analyze.py --input-name="codec-constraints-gm"

# Constraints + ASR
python $CDIR/main_train_asr.py
python $CDIR/main_train_codec.py --with-envelope --with-asr --output-name="codec-constraints-asr"
python $CDIR/main_analyze.py --input-name="codec-constraints-asr"

# Constraints + SR
python $CDIR/main_train_sr.py
python $CDIR/main_train_codec.py --with-envelope --with-sr --output-name="codec-constraints-sr"
python $CDIR/main_analyze.py --input-name="codec-constraints-sr"

# Constraints + ASR + GM
python $CDIR/main_train_codec.py --with-envelope --with-asr --with-gm --output-name="codec-constraints-asr-gm"
python $CDIR/main_analyze.py --input-name="codec-constraints-asr-gm"

echo 'All done.'


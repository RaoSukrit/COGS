#!/bin/bash -l

# The directory that contains the COGS dataset reformatted into OpenNMT input format.
export DATA_CONFIG=$1
export OPENNMT_DIR=$2

python $OPENNMT_DIR/build_vocab.py \
        -config $DATA_CONFIG \
        -n_sample 25_000

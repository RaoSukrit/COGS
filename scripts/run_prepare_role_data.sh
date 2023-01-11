#!/bin/bash -l

export DATA_PATH=$1                            # OpenNMT-preprocessed data path
export OPENNMT_PATH=$2                         # OpenNMT path
export EXAMPLES=1_example                      # Number of exposure examples (1 or 100)
export SAVE_PATH=${OPENNMT_PATH}/tf_checkpoints  # Save path for checkpoints
export SAVE_NAME=${EXAMPLES}_gru_uni_att_2layers_no_if_hidden512           # Checkpoint name
export LOG_PATH=${OPENNMT_PATH}/logs           # Log path
export PRED_PATH=${OPENNMT_PATH}/preds         # Predictions path
export SEED=42                                  # Random seed
export CUDA_VISIBLE_DEVICES=0                    # GPU machine number
export MODEL_SAVE_DIR=../role_data/cogs_${SAVE_NAME}/saved_models # Location of exported cogs decoder weights
export EMBD_OUTPUT_SAVE_NAME=../role_data/cogs_${SAVE_NAME}/cogs_${SAVE_NAME}_seed${SEED}_embd

mkdir $LOG_PATH
mkdir $PRED_PATH
mkdir -p ${MODEL_SAVE_DIR}

# Inference
# for SPLIT in dev
for SPLIT in gen train dev test
do
    python $OPENNMT_PATH/prepare_role_data.py -model $SAVE_PATH/${SAVE_NAME}/s${SEED}_best.pt \
                                              -src $DATA_PATH/${SPLIT}_source.txt \
                                              -tgt $DATA_PATH/${SPLIT}_target.txt \
                                              -output ${EMBD_OUTPUT_SAVE_NAME}.data_from_${SPLIT} \
                                              -model_save_dir ${MODEL_SAVE_DIR} \
                                              -model_prefix ${SAVE_NAME} \
                                              -verbose -shard_size 0 \
                                              -gpu 0 -batch_size 1 \
                                              -rnn_type GRU -layers 2
                                               #-debug

done

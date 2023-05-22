#!/bin/bash -l

export DATA_PATH=$1                            # OpenNMT-preprocessed data path
export OPENNMT_PATH=$2                         # OpenNMT path
export SEED=$3                                  # Random seed
export EXAMPLES=1_example                      # Number of exposure examples (1 or 100)
export SAVE_PATH=${OPENNMT_PATH}/tf_checkpoints  # Save path for checkpoints
export SAVE_NAME=seed_${SEED}_${EXAMPLES}_gru_uni_att_1layers_no_if_hidden512          # Checkpoint name
# export SAVE_NAME=${EXAMPLES}_gru_uni_no_att_1layers_no_if_hidden512
export LOG_PATH=${OPENNMT_PATH}/logs           # Log path
export PRED_PATH=${OPENNMT_PATH}/preds         # Predictions path
export CUDA_VISIBLE_DEVICES=-1                    # GPU machine number

mkdir $LOG_PATH
mkdir $PRED_PATH

# Inference
for SPLIT in dev
# for SPLIT in gen train dev test
do
    python $OPENNMT_PATH/test_prepare_role_data.py -model $SAVE_PATH/${SAVE_NAME}/s${SEED}_best.pt \
                                              -src $DATA_PATH/${SPLIT}_source.txt \
                                              -tgt $DATA_PATH/${SPLIT}_target.txt \
                                              -model_prefix ${SAVE_NAME} \
                                              -verbose -shard_size 0 \
                                              -gpu ${CUDA_VISIBLE_DEVICES} -batch_size 1 \
                                              -rnn_type GRU \
                                              -layers 1 -attn_model
                                               #-debug

done

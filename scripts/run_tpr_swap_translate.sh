#!/bin/bash -l

export DATA_PATH=$1                            # OpenNMT-preprocessed data path
export OPENNMT_PATH=$2                         # OpenNMT path
export EXAMPLES=1_example                      # Number of exposure examples (1 or 100)
export SAVE_PATH=${OPENNMT_PATH}/tf_checkpoints  # Save path for checkpoints
export SAVE_NAME=${EXAMPLES}_lstm_uni_no_att_1layers           # Checkpoint name
export LOG_PATH=${OPENNMT_PATH}/logs           # Log path
export PRED_PATH=${OPENNMT_PATH}/preds         # Predictions path
export SEED=1                                  # Random seed
export CUDA_VISIBLE_DEVICES=-1                  # GPU machine number
export TPR_CONFIG=../src/OpenNMT-py/config/config-tpr-encoder.yml
# export TPR_DATA_PREFIX=

mkdir $LOG_PATH
mkdir $PRED_PATH

                                    #   -tpr_data_prefix=$TPR_DATA_PREFIX
# Inference
#for SPLIT in gen test dev
#for SPLIT in train train_100 test dev gen
for SPLIT in test
do
    python $OPENNMT_PATH/tpr_swap_translate.py -nmt_model $SAVE_PATH/${SAVE_NAME}/s${SEED}_best.pt \
                                      -src $DATA_PATH/${SPLIT}_source.txt \
                                      -tgt $DATA_PATH/${SPLIT}_target.txt \
				      -output ../role_data/cogs_${SAVE_NAME}_${SEED}_1_embd.data_from_${SPLIT} \
                                      -verbose -shard_size 0 \
                                      -gpu -1 -batch_size 1 \
                                      -tpr_config ${TPR_CONFIG} \
                                      --max_length 2000 \

done

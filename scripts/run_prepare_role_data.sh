#!/bin/bash -l

export DATA_PATH=$1                            # OpenNMT-preprocessed data path
export OPENNMT_PATH=$2                         # OpenNMT path
export SEED=$3                                  # Random seed
export EXAMPLES=1_example                      # Number of exposure examples (1 or 100)
export SAVE_PATH=${OPENNMT_PATH}/tf_checkpoints/1_layer_with_attn_new_vocab_debugging  # Save path for checkpoints
export SAVE_NAME=seed_${SEED}_${EXAMPLES}_gru_uni_att_1layers_no_if_hidden512          # Checkpoint name
# export SAVE_NAME=${EXAMPLES}_gru_uni_no_att_1layers_no_if_hidden512
export LOG_PATH=${OPENNMT_PATH}/logs/1_layer_with_attn_new_vocab_debugging/seed_${SEED}          # Log path
export PRED_PATH=${OPENNMT_PATH}/preds/1_layer_with_attn_new_vocab_debugging/seed_${SEED}         # Predictions path
export CUDA_VISIBLE_DEVICES=-1                # GPU machine number
export MODEL_SAVE_DIR=../role_data/1_layer_with_attn_new_vocab/cogs_${SAVE_NAME}/saved_models # Location of exported cogs decoder weights
export EMBD_OUTPUT_SAVE_NAME=../role_data/1_layer_with_attn_new_vocab/cogs_${SAVE_NAME}/cogs_${SAVE_NAME}_embd

mkdir $LOG_PATH
mkdir $PRED_PATH
mkdir -p ${MODEL_SAVE_DIR}

# Inference
# for SPLIT in dev
for SPLIT in gen dev test
do
    python $OPENNMT_PATH/prepare_role_data.py -model $SAVE_PATH/${SAVE_NAME}/s${SEED}_best.pt \
                                              -src $DATA_PATH/new_vocab_exp_attn_role/$EXAMPLES/${SPLIT}_source.txt \
                                              -tgt $DATA_PATH/new_vocab_exp_attn_role/$EXAMPLES/${SPLIT}_target.txt \
                                              -output ${EMBD_OUTPUT_SAVE_NAME}.data_from_${SPLIT} \
                                              -model_save_dir ${MODEL_SAVE_DIR} \
                                              -model_prefix ${SAVE_NAME} \
                                              -verbose -shard_size 0 \
                                              -gpu ${CUDA_VISIBLE_DEVICES} -batch_size 1 \
                                              -rnn_type GRU \
                                              -layers 1 -attn_model  \
                                              -data_split ${SPLIT}
                                               #-debug

done

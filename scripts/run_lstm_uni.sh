#!/bin/bash -l

export DATA_PATH=$1                            # OpenNMT-preprocessed data path
export OPENNMT_PATH=$2                         # OpenNMT path
export EXAMPLES=1_example                      # Number of exposure examples (1 or 100)
export SAVE_PATH=${OPENNMT_PATH}/tf_checkpoints  # Save path for checkpoints
export SAVE_NAME=${EXAMPLES}_lstm_uni_no_att_1layers_no_if_proj          # Checkpoint name
export LOG_PATH=${OPENNMT_PATH}/logs           # Log path
export PRED_PATH=${OPENNMT_PATH}/preds         # Predictions path
export SEED=42                                  # Random seed
export CUDA_VISIBLE_DEVICES=-1                  # GPU machine number
export LOAD_CHECKPOINT=/scratch/str8775/conda_env/compositional-generalisation/COGS/src/OpenNMT-py/tf_checkpoints/1_example_gru_uni_no_att_1layers_no_if/s42_best.pt                      # load saved checkpoint

mkdir $LOG_PATH
mkdir $PRED_PATH

# Training
python -u $OPENNMT_PATH/train.py -config ./config/build_vocab_1_example.yml -save_model $SAVE_PATH/${SAVE_NAME}/s$SEED \
	-src_vocab $DATA_PATH/$EXAMPLES/${EXAMPLES}.vocab.src \
	-tgt_vocab $DATA_PATH/$EXAMPLES/${EXAMPLES}.vocab.tgt \
	-layers 1 -rnn_size 512 -word_vec_size 512 \
	-encoder_type rnn -decoder_type rnn -rnn_type LSTM \
	-input_feed 0 -global_attention none \
	-train_steps 30000  -max_generator_batches 2 -dropout 0.0 \
	-batch_size 256 -batch_type sents -normalization sents  -accum_count 4 \
	-optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 4000 -learning_rate 2 \
       -max_grad_norm 5 -param_init 0  \
	-valid_steps 500 -save_checkpoint_steps 500 \
	-early_stopping 15 --early_stopping_criteria accuracy \
	-world_size 1 -gpu_ranks 0 -seed $SEED --log_file ${LOG_PATH}/${SAVE_NAME}_s${SEED}.log \
	# -train_from ${LOAD_CHECKPOINT}



# Inference
# for SPLIT in gen test dev
# for SPLIT in dev
# do
#     python $OPENNMT_PATH/translate.py -model $SAVE_PATH/${SAVE_NAME}/s${SEED}_best.pt \
#                                       -src $DATA_PATH/${SPLIT}_source.txt \
#                                       -tgt $DATA_PATH/${SPLIT}_target.txt \
#                                       -output ${PRED_PATH}/${SPLIT}_pred_${SAVE_NAME}_s${SEED}.txt \
#                                       -verbose -shard_size 0 \
#                                       -gpu 0 -batch_size 128  > ${PRED_PATH}/${SPLIT}_pred_${SAVE_NAME}_s${SEED}_log.txt

#     paste $DATA_PATH/${SPLIT}_source.txt $DATA_PATH/${SPLIT}_target.txt ${PRED_PATH}/${SPLIT}_pred_${SAVE_NAME}_s${SEED}.txt > ${PRED_PATH}/${SPLIT}_pred_${SAVE_NAME}_s${SEED}.tsv
# done

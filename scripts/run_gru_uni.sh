#!/bin/bash -l

export DATA_PATH=$1                            # OpenNMT-preprocessed data path
export OPENNMT_PATH=$2                         # OpenNMT path
export SEED=$3                                  # Random seed
export EXAMPLES=1_example                      # Number of exposure examples (1 or 100)
export SAVE_PATH=${OPENNMT_PATH}/tf_checkpoints/1_layer_with_attn_new_vocab_debugging  # Save path for checkpoints
export SAVE_NAME=seed_${SEED}_${EXAMPLES}_gru_uni_att_1layers_no_if_hidden512          # Checkpoint name
# export SAVE_NAME=${EXAMPLES}_gru_uni_no_att_1layers_no_if_hidden512
export LOG_PATH=${OPENNMT_PATH}/logs/1_layer_with_attn_new_vocab_debugging/gru_seed_${SEED}          # Log path
export PRED_PATH=${OPENNMT_PATH}/preds/1_layer_with_attn_new_vocab_debugging/gru_seed_${SEED}         # Predictions path
export CUDA_VISIBLE_DEVICES=0                # GPU machine number
# export LOAD_CHECKPOINT=/scratch/str8775/conda_env/compositional-generalisation/COGS/src/OpenNMT-py/tf_checkpoints/1_example_gru_uni_att_1layers_no_if_hidden512/s42_step_65000.pt # load saved checkpoint

mkdir -p $LOG_PATH
mkdir -p $PRED_PATH

# -config ./config/build_vocab_1_example.yml \
	# -src_vocab $DATA_PATH/$EXAMPLES/${EXAMPLES}.vocab.src \
	# -tgt_vocab $DATA_PATH/$EXAMPLES/${EXAMPLES}.vocab.tgt \

# # Training
# python -u $OPENNMT_PATH/train.py \
#  	-save_model $SAVE_PATH/${SAVE_NAME}/s$SEED \
#  	-src_vocab $DATA_PATH/new_vocab_exp_attn_role/$EXAMPLES/source_vocab.txt \
#  	-tgt_vocab $DATA_PATH/new_vocab_exp_attn_role/$EXAMPLES/target_vocab.txt \
#  	-data $DATA_PATH/new_vocab_exp_attn_role/$EXAMPLES/data.yml \
#  	-layers 1 -rnn_size 512 -word_vec_size 512 \
#  	-encoder_type rnn -decoder_type rnn -rnn_type GRU \
#  	-input_feed 0 -global_attention dot \
# 	-train_steps 80000  -max_generator_batches 2 -dropout 0.1 \
#  	-batch_size 512 -batch_type sents -normalization sents  -accum_count 4 \
#         -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 4000 -learning_rate 2 \
#         -max_grad_norm 5 -param_init 0  \
#  	-valid_steps 1000 -save_checkpoint_steps 1000 \
#  	-early_stopping 10 --early_stopping_criteria accuracy \
#  	-world_size 1 -gpu_ranks 0 -seed $SEED --log_file ${LOG_PATH}/${SAVE_NAME}_s${SEED}.log
# # # # 	# -train_from ${LOAD_CHECKPOINT}



# Inference
#for SPLIT in gen
# # # #
for SPLIT in gen test train dev
# # for SPLIT in dev

do
   python $OPENNMT_PATH/translate.py -model $SAVE_PATH/${SAVE_NAME}/s${SEED}_best.pt \
                                     -src $DATA_PATH/new_vocab_exp_attn_role/$EXAMPLES/${SPLIT}_source.txt \
                                     -tgt $DATA_PATH/new_vocab_exp_attn_role/$EXAMPLES/${SPLIT}_target.txt \
                                     -output ${PRED_PATH}/${SPLIT}_pred_${SAVE_NAME}_s${SEED}.txt \
                                     -verbose -shard_size 0 \
                                     -gpu ${CUDA_VISIBLE_DEVICES} -batch_size 512  > "${PRED_PATH}/${SPLIT}_pred_${SAVE_NAME}_s${SEED}_log.txt"

   paste $DATA_PATH/${SPLIT}_source.txt $DATA_PATH/${SPLIT}_target.txt ${PRED_PATH}/${SPLIT}_pred_${SAVE_NAME}_s${SEED}.txt > ${PRED_PATH}/${SPLIT}_pred_${SAVE_NAME}_s${SEED}.tsv
done

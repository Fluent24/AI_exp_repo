#!/bin/bash

LANG="en"
LABEL_TYPE1="pron"
DIR_LIST="/mnt/f/fluent/AI_exp_repo/datasets_list"
MLP_HIDDEN=64
EPOCHS=200
PATIENCE=20
BATCH_SIZE=1024
DIR_MODEL="model_compare_3arch"
PNAME="model_compare_report_01"

for MODEL_TYPE in "mlp" "cnn+lstm" "transformer"; do
    for LABEL_TYPE2 in "articulation" "prosody"; do
        for MLP_HIDDEN in 64 128 256; do
            if [ "$MODEL_TYPE" == "transformer" ]; then
                for NUM_LAYER in 1; do
                    for NUM_HEAD in  2 4 8; do
                        RUN_NAME="trans_${MLP_HIDDEN}_${NUM_LAYER}_${NUM_HEAD}"
                        python compare_models.py \
                            --lang="$LANG" \
                            --label_type1="$LABEL_TYPE1" \
                            --label_type2="$LABEL_TYPE2" \
                            --dir_list="$DIR_LIST" \
                            --mlp_hidden="$MLP_HIDDEN" \
                            --epochs="$EPOCHS" \
                            --patience="$PATIENCE" \
                            --batch_size="$BATCH_SIZE" \
                            --dir_model="$DIR_MODEL" \
                            --model_type="$MODEL_TYPE" \
                            --num_layer="$NUM_LAYER" \
                            --num_head="$NUM_HEAD"\
                            --p_name="$PNAME"\
                            --run_name="$RUN_NAME"
                    done
                done
            elif [ "$MODEL_TYPE" == "cnn+lstm" ]; then
                for NUM_LAYER in 1; do
                    RUN_NAME="c+l_${MLP_HIDDEN}_${NUM_LAYER}"
                    python compare_models.py \
                        --lang="$LANG" \
                        --label_type1="$LABEL_TYPE1" \
                        --label_type2="$LABEL_TYPE2" \
                        --dir_list="$DIR_LIST" \
                        --mlp_hidden="$MLP_HIDDEN" \
                        --epochs="$EPOCHS" \
                        --patience="$PATIENCE" \
                        --batch_size="$BATCH_SIZE" \
                        --dir_model="$DIR_MODEL" \
                        --model_type="$MODEL_TYPE" \
                        --num_layer="$NUM_LAYER"\
                        --p_name="$PNAME"\
                        --run_name="$RUN_NAME"
                done
            else
                for NUM_LAYER in 1 2 4; do
                    RUN_NAME="mlp_${MLP_HIDDEN}_${NUM_LAYER}"
                    python compare_models.py \
                        --lang="$LANG" \
                        --label_type1="$LABEL_TYPE1" \
                        --label_type2="$LABEL_TYPE2" \
                        --dir_list="$DIR_LIST" \
                        --mlp_hidden="$MLP_HIDDEN" \
                        --num_layer="$NUM_LAYER" \
                        --epochs="$EPOCHS" \
                        --patience="$PATIENCE" \
                        --batch_size="$BATCH_SIZE" \
                        --dir_model="$DIR_MODEL" \
                        --model_type="$MODEL_TYPE"\
                        --p_name="$PNAME"\
                        --run_name="$RUN_NAME"
                done
            fi
        done
    done
done

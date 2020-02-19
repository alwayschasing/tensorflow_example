#!/bin/bash

now=`date -d "now" "+%Y-%m-%d %H:%M:%S"`
echo "[$now]Start" 
rootdir="/search/odin/workspace/querySemantic"

python run_model.py \
    --input_file=$rootdir/data/transformerData/google_part \
    --output_dir=$rootdir/transformer_model/output \
    --max_seq_length=64 \
    --max_predictions_per_seq=5 \
    --embedding_table_file=$rootdir/data/transformerData/vocab_vec.npy \
    --embedding_table_trainable=False \
    --num_train_steps=70 \
    --vocab_size=1513035 \
    --vocab_vec_size=200 \
    --do_train=True
#--init_checkpoint=None \

now=`date -d "now" "+%Y-%m-%d %H:%M:%S"`
echo "[$now]Finish"

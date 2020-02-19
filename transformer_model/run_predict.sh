#!/bin/bash

echo "[$now]Start" 
rootdir="/search/odin/workspace/querySemantic"

input_file="predict.txt"
init_checkpoint="train_output_5/"
vocab_file="/search/odin/workspace/querySemantic/data/transformerData/vocab"
vocab_vec_file="/search/odin/workspace/querySemantic/data/transformerData/vocab_vec.npy"
predict_output="/search/odin/workspace/querySemantic/data/transformerData/predict_output"

python run_model.py \
    --input_file=$input_file \
    --max_seq_length=64 \
    --vocab_file=$vocab_file \
    --vocab_vec_file=$vocab_vec_file \
    --do_predict=True \
    --init_checkpoint=$init_checkpoint \
    --embedding_table_file=$vocab_vec_file \
    --vocab_size=1513035 \
    --vocab_vec_size=200 \
    --output_dir=$predict_output


now=`date -d "now" "+%Y-%m-%d %H:%M:%S"`
echo "[$now]Finish"

#!/bin/bash

w2v_file="/search/odin/workspace/querySemantic/data/wordRepresentation/click2vec/retrain_word2click_vectors_v2"
addWord_file="/search/odin/workspace/querySemantic/transformer_model/addWord.txt"
saved_vocab_file="/search/odin/workspace/querySemantic/data/transformerData/vocab"
saved_vocab_vec_file="/search/odin/workspace/querySemantic/data/transformerData/vocab_vec"

python buildVocab.py \
    --w2v_file $w2v_file \
    --addWord_file $addWord_file \
    --vocab_file $saved_vocab_file \
    --vocab_vec_file $saved_vocab_vec_file 


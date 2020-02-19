#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tokenization

vocab_file = "/search/odin/workspace/querySemantic/data/transformerData/vocab"
vocab_vec_file="/search/odin/workspace/querySemantic/data/transformerData/vocab_vec.npy"
tokenizer = tokenization.Tokenizer(vocab_file,vocab_vec_file)


def test():
    tokens = ["","","","","",""]
    ids = tokenizer.convert_tokens_to_ids(tokens)

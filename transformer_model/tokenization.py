#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
# from segmentor import segmentor
from util import CodeUtil

class Tokenizer(object):
    """Runs basic tokenization (punctuation spliting, lower casing, etc.)."""
    # word_start_index: the start index of words
    def __init__(self,vocab_file,vocab_vec_file,vec_size=200):
        self.vocab = load_vocab_file(vocab_file)
        self.vectors = load_vocab_vec_file(vocab_vec_file)
        self.vocab_size = len(self.vocab)
        self.token2id = {}
        for i in xrange(self.vocab_size):
            self.token2id[self.vocab[i]] = i

        # self.segmenter = segmentor.Segmentor()
    
    # def tokenize(self, text):
        # """Tokenizes a piece of text"""
        # tokens = self.segmenter.segment(text)
        # return tokens

    def convert_tokens_to_ids(self, tokens):
        ids = []
        for tok in tokens:
            if tok in self.token2id:
                ids.append(self.token2id[tok])
            else:
                ids.append(self.token2id[CodeUtil.UTFNormalize('[UNK]')])
        return ids

    def convert_ids_to_tokens(self,ids):
        tokens = []
        for id in ids:
            tokens.append(self.vocab[id])
        return tokens



def load_vocab_file(vocab_file):
    words = []
    with open(vocab_file,'r') as rfp:
        for line in rfp:
            word = line.rstrip()
            words.append(word)
    return words

def load_vocab_vec_file(vocab_vec_file,fmt="binary"):
    if fmt == "binary":
        vocab_vec = np.load(vocab_vec_file)
        return vocab_vec
    else:
        raise Exception("load vocab_vec_file error, fmt error")


if __name__ == "__main__":
    vocab_file = "/search/odin/workspace/querySemantic/data/transformerData/vocab"
    vocab_vec_file="/search/odin/workspace/querySemantic/data/transformerData/vocab_vec.npy"
    tokenizer = Tokenizer(vocab_file,vocab_vec_file)
    

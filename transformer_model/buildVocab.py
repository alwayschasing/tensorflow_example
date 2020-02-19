#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gensim
import numpy as np
from util import CodeUtil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-w","--w2v_file",type=str, help="word2vec model file")
parser.add_argument("-s","--addWord_file", type=str, help="added special word file")
parser.add_argument("-b","--vocab_file",type=str,help="saved vocab file")
parser.add_argument("-v","--vocab_vec_file",type=str,help="saved vocab vec file")

def buildVocabFromWord2VecFile(word2vec_model_file,addWord_file,vocab_file,vocab_vec_file):
    addWords = loadAddWords(addWord_file,encoding='gb18030')
    model = gensim.models.Word2Vec.load(word2vec_model_file)
    words_count = len(model.wv.index2word)
    hidden_size = len(model.wv[model.wv.index2word[0]])
    vocab = addWords + model.wv.index2word
    with open(vocab_file,'w') as wfp:
        for w in vocab:
            wfp.write(w + '\n')

    vocab_vec = np.random.normal(0, 0.1,[len(addWords)+words_count,hidden_size]) 
    st = len(addWords)
    for i in range(words_count):
        # vocab_vec = np.vstack((vocab_vec,model.wv[model.wv.index2word[i]]))
        vocab_vec[st+i] = model.wv[model.wv.index2word[i]]
    np.save(vocab_vec_file,vocab_vec)

def loadAddWords(addWords_file,encoding='utf-8'):
    words = []
    with open(addWords_file) as fp:
        for line in fp.readlines():
            word = line.strip()
            if encoding == 'gbk' or encoding == 'gb18030':
                word = CodeUtil.GBKNormalize(word)
            else:
                word = CodeUtil.UTFNormalize(word)
            words.append(word)
    return words


def main():
    args = parser.parse_args()
    word2vec_model_file = args.w2v_file
    addWord_file = args.addWord_file
    vocab_file = args.vocab_file
    vocab_vec_file = args.vocab_vec_file
    buildVocabFromWord2VecFile(word2vec_model_file,addWord_file,vocab_file,vocab_vec_file)


if __name__ == "__main__":
    main()

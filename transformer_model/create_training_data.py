#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import collections
import tokenization
import random
from util import CodeUtil
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("-i","--input_files",type=str, help="input files")
# parser.add_argument("-o","--output_files",type=str, help="output files")
# parser.add_argument("-l","--max_seq_length",type=str, help="max_seq_length")
# parser.add_argument("-l","--max_seq_length",type=str, help="max_seq_length")


def create_training_data_from_files(input_files,
                                    output_files,
                                    max_seq_length,
                                    masked_lm_prob,
                                    max_predictions_per_seq,
                                    tokenizer,
                                    rng):

    tf.logging.info("start create training data")
    writers = []
    for output_file in output_files:
        print output_file
        writers.append(tf.python_io.TFRecordWriter(output_file))
    writer_index = 0

    k_count = 0
    for input_file in input_files:
        with tf.gfile.GFile(input_file,'r') as reader:
            tf.logging.info("create training data from %s" % (input_file))
            # last_query_items = []
            last_title_items = []
            while True:
                line = reader.readline()
                if not line:
                    break
                # query_items\ttitle_items
                items = line.strip().split('\t')
                # seperated by ','
                query_items = items[0].split(',')
                title_items = items[1].split(',')



                positive_example = createExample(query_items,title_items,1,max_seq_length,masked_lm_prob,max_predictions_per_seq,tokenizer,rng,k_count)
                if len(last_title_items) > 0:
                    negative_example = createExample(query_items,last_title_items,0,max_seq_length,masked_lm_prob,max_predictions_per_seq,tokenizer,rng,k_count)

                # tokens = [CodeUtil.UTFNormalize("[CLS]")] + query_items + [CodeUtil.UTFNormalize("[SEP]")] + title_items
                # n_query_item = len(query_items)
                # n_title_item = len(title_items)

                # input_mask = [0] * max_seq_length
                # segment_ids = [0] * max_seq_length

                # for i in range(n_query_item + 2 + n_title_item):
                    # input_mask[i] = 1

                # for i in range(n_query_item + 2):
                    # segment_ids[i] = 1

                # for i in range(n_query_item + 2, n_query_item + 2 + n_title_item):
                    # segment_ids[i] = 0
                
                # (tokens, masked_lm_ids, masked_lm_positions) = create_masked_lm_predictions(tokens,
                                                                                   # masked_lm_prob,
                                                                                   # max_predictions_per_seq,
                                                                                   # rng)
                # masked_lm_weights = [1.0] * len(masked_lm_ids) 
                # while len(masked_lm_positions) < max_predictions_per_seq:
                    # masked_lm_positions.append(0)
                    # masked_lm_ids.append(0)
                    # masked_lm_weights.append(0)

                # input_ids = tokenizer.convert_tokens_to_ids(tokens)

                # features = collections.OrderedDict()
                # features["input_ids"] = create_int_feature(input_ids)
                # features["input_mask"] = create_int_feature(input_mask)
                # features["segment_ids"] = create_int_feature(segment_ids)
                # features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
                # features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
                # features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
                # features["labels"] = create_int_feature([label])

                # tf_example = tf.train.Example(features=tf.train.Features(features=features))
                # writers[writer_index].write(tf_example.SerializeToString())
                writers[writer_index].write(positive_example.SerializeToString())
                if len(last_title_items) > 0:
                    writers[writer_index].write(negative_example.SerializeToString())
                writer_index = (writer_index + 1) % len(writers)
                # if k_count < 20:
                    # tf.logging.info("*** Example ***")
                    # tf.logging.info("tokens:%s" % " ".join(
                        # [tokenization.printable_text(x) for x in tokens]))
                    # for feature_name in features.keys():
                        # feature = features[feature_name]
                        # values = []
                        # if feature.int64_list.value:
                            # values = feature.int64_list.value
                        # elif feature.float_list.value:
                            # values = feature.float_list.value
                        # tf.logging.info(
                            # "%s: %s"%(feature_name, " ".join([str(x) for x in values])))
                k_count += 1
                last_title_items = title_items
                # last_query_items = query_items
            tf.logging.info("finish training data from %s, now example nums:%d" % (input_file,k_count))
    tf.logging.info("finish all training data creating, total example nums:%d" % (k_count))
    for writer in writers:
        writer.close()

def createExample(tokens_a,tokens_b,label,max_seq_length,masked_lm_prob,max_predictions_per_seq,tokenizer,rng,k_count):
    tokens = [CodeUtil.UTFNormalize("[CLS]")] + tokens_a + [CodeUtil.UTFNormalize("[SEP]")] + tokens_b
    n_a = len(tokens_a)
    n_b = len(tokens_b)
    input_mask = [0] * max_seq_length
    segment_ids = [0] * max_seq_length
    for i in range(n_a + 2 + n_b):
        input_mask[i] = 1

    for i in range(n_a + 2):
        segment_ids[i] = 1

    for i in range(n_a + 2, n_a + 2 + n_b):
        segment_ids[i] = 0

    (tokens, masked_lm_labels, masked_lm_positions) = create_masked_lm_predictions(tokens,
                                                                       masked_lm_prob,
                                                                       max_predictions_per_seq,
                                                                       rng)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    masked_lm_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)
    masked_lm_weights = [1.0] * len(masked_lm_ids) 

    assert len(input_ids) <= max_seq_length
    while len(input_ids) < max_seq_length:
        input_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    while len(masked_lm_positions) < max_predictions_per_seq:
        masked_lm_positions.append(0)
        masked_lm_ids.append(0)
        masked_lm_weights.append(0)


    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(input_mask)
    features["segment_ids"] = create_int_feature(segment_ids)
    features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
    features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
    features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
    features["labels"] = create_int_feature([label])
    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    if k_count < 20:
        tf.logging.info("*** Example ***")
        tf.logging.info("tokens:%s" % " ".join(tokens))
        # tf.logging.info("tokens:%s" % " ".join(
            # [tokenization.printable_text(x) for x in tokens]))
        for feature_name in features.keys():
            feature = features[feature_name]
            values = []
            if feature.int64_list.value:
                values = feature.int64_list.value
            elif feature.float_list.value:
                values = feature.float_list.value
            tf.logging.info(
                "%s: %s"%(feature_name, " ".join([str(x) for x in values])))
    return tf_example

def create_masked_lm_predictions(tokens,masked_lm_prob,max_predictions_per_seq,rng):
    cand_indexes = []
    for (i,token) in enumerate(tokens):
        if token == CodeUtil.UTFNormalize("[SEP]") or token == CodeUtil.UTFNormalize("[CLS]"):
            continue
        cand_indexes.append(i)

    rng.shuffle(cand_indexes)
    output_tokens = list(tokens)
    masked_lm = collections.namedtuple("masked_lm", ["index", "label"])

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    for index in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if index in covered_indexes:
            continue
        covered_indexes.add(index)

        masked_token = None
        if rng.random() > 0.8:
            masked_token = CodeUtil.UTFNormalize("[MASK]")
        else:
            masked_token = tokens[index]
        
        output_tokens[index] = masked_token
        masked_lms.append(masked_lm(index=index, label=tokens[index]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)
            
    return (output_tokens, masked_lm_labels, masked_lm_positions)
    

def generate_training_file(input_files, training_files):
    pass

def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature
                
def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature

def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    input_files = ["/search/odin/workspace/spider_data/clean_data/google_QueryAndTitle_5"]
    output_files = ["/search/odin/workspace/querySemantic/data/transformerData/google_part_5"]
    max_seq_length = 64
    masked_lm_prob = 0.2
    max_predictions_per_seq = 5
    vocab_file = "/search/odin/workspace/querySemantic/data/transformerData/vocab"
    vocab_vec_file = "/search/odin/workspace/querySemantic/data/transformerData/vocab_vec.npy"
    tokenizer = tokenization.Tokenizer(vocab_file,vocab_vec_file)
    rng = random.Random(12345)
    create_training_data_from_files(input_files,
                                    output_files,
                                    max_seq_length,
                                    masked_lm_prob,
                                    max_predictions_per_seq,
                                    tokenizer,
                                    rng)

if __name__ == "__main__":
    main()

                

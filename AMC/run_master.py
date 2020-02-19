#!/usr/bin/env python
# -*- coding:utf-8 -*-
import tensorflow as tf
import modeling
import optimization
flags = tf.flags
FLAGS = flags.FLAGS

class InputExample(object):
    """A single training/test example."""
    def __init__(self,query,keys,image):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            query: string, query text 
            Keys: list strings. The untokenized Keys of every image.
            image: 1 relative image and n unrelative image
        """
        self.query = query
        self.Keys = keys
        self.image = image

class InputFeatures(object):
    """A single set of features of data.
    Args:
        query_vec: vector representation of query_vec
        key_vecs: vector representation of every key set of image
        image_vecs: vector representation of every image  
    """
    def __init__(self,query_vec,key_vecs,image_vecs):
        self.query_vec = query_vec
        self.key_vecs = key_vecs
        self.image_vecs = image_vecs

class DataProcessor(object):
    """Base class for data converters for AMT data sets."""
    def get_train_examples(self, data_dir):
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        raise NotImplementedError

    def get_test_example(self, data_dir):
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file,"r") as f:
            reader = csv.reader(f,delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


## Required parameters

def model_fn_builder(config):
    """Return model_fn for Estimator"""
    def model_fn(features,labels,mode,params):
        """The model_fn for Estimator"""
        input_q = features["input_q"] # query feature vector
        input_K = features["input_K"] # Key set Matrix
        input_v = features["input_v"] # image visual feature vector
        input_labels = features["input_labels"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        model = modeling.AMT(
            config = config,
            is_trainging = is_training, 
            scope = "AMT",
            input_q = input_q,
            input_K = input_K,
            input_v = input_v
            )
        loss = model.loss
        q_doc_rank = model.get_predict()
        output_spec = None
        scaffold_fn = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer()
            output_spec = tf.estimator.EstimatorSpec(
                mode = mode,
                loss = loss,
                train_op = train_op,
                scaffold_fn = scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn():
                return 0
        else:
            output_spec = tf.estimator.EstimatorSpec(
                mode = mode,
                predictions = q_doc_rank,
                scaffold_fn = scaffold_fn)
        return output_spec
    return model_fn

def file_based_convert_examples_to_features(examples,label_list,output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""
    writer = tf.python_io.TFRecordWriter(output_file)
    for(ex_index, example) in enumerate(examples):
        if ex_index%10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list)
        
        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f
        features = {
            "":tf.train.Feature(int64_list=tf.train.Int64List(value=list(values))),
        }

        tf_example = tf.train.Example(features=tf.train.Features(features=features))
        writer.write(tf_example.SerializeToString())

def file_based_input_fn_builder(input_file):
    """create an `input_fn` closure to be passed to Estimator."""
    # 存放解析自TFRecord文件的数据
    name_to_features = {
        "input_q":tf.FixedLenFeature([shape],tf.int64),
        "input_K":tf.FixedLenFeature([],tf.float32),
        "input_v":tf.FixedLenFeature([],tf.float32),
    }

    def _decode_record(record,name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record,name_to_features)

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size = 100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record:_decode_record(record, name_to_features),
                batch_size = batch_size,
                drop_remainder=drop_remainder))
        return d
    return input_fn


def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    if not FLAGS.do_train and not FLAGS.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    config = {}
    model_fn = model_fn_builder()
    estimator = tf.estimator.Estimator(
        model_fn = model_fn,
        )

    if FLAGS.do_train:
        train_input_fn = file_based_input_fn_builder()
        estimator.train(input_fn = train_input_fn,
                        max_steps = num_train_steps)
    if FLAGS.do_eval:
        pass

    if FLAGS.do_predict:
        pass

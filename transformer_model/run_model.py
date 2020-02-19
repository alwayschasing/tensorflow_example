#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import modeling
import optimization
import tokenization
import random
from util import CodeUtil

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "input_file", None,
    "Input TF example files(can be a glob or comma separated).")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")
flags.DEFINE_string(
    "embedding_table_file",None,
    "The embedding table file, numpy data.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Intial checkpoints (usually from a pre-trained mdoel).")
flags.DEFINE_string(
    "vocab_file",None,
    "vocab file when predict")
flags.DEFINE_string(
    "vocab_vec_file", None,
    "vocab vec file when predict")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after tokenization")

flags.DEFINE_integer(
    "max_predictions_per_seq",20,
    "Maximum number of masked LM predictions per seqence. ")
flags.DEFINE_integer(
    "vocab_size",0,
    "vocab_size")

flags.DEFINE_integer(
    "vocab_vec_size",200,
    "word vec size of vocab")

flags.DEFINE_bool(
    "embedding_table_trainable", False,
    "Whether to train embedding table.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")
flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")
flags.DEFINE_integer("eval_batch_size", 32, "Total batch size for eval.")
flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")
flags.DEFINE_integer("num_train_steps", 10, "Number of training steps.")
flags.DEFINE_integer("num_warmup_steps", 10, "Number of warmup steps.")
flags.DEFINE_integer("save_checkpoints_steps",1000,
                     "How often to save the model checkpoint.")
flags.DEFINE_integer("random_seed",12345,"random seed.")
flags.DEFINE_integer("neg_sample_num",5,"negative sample number.")

flags.DEFINE_bool("do_predict", False, "Whether to run predict.")

def model_fn_builder(config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps,
                     use_one_hot_embeddings=False,
                     embedding_table=None,
                     embedding_table_trainable=False,
                     rng=random.Random(12345)):
    # num_labels = config.num_labels
    # hidden_size = config.hidden_size
    """Returns `model_fn` closure for Estimator."""

    def model_fn(features, labels, mode, params):
        """The `model_fn` for Estimator.
        """

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info(" name = %s, shape = %s" % (name, features[name].shape))

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]

        # if is_training:
        if mode != tf.estimator.ModeKeys.PREDICT:
            masked_lm_ids = features["masked_lm_ids"]
            masked_lm_positions = features["masked_lm_positions"]
            masked_lm_weights = features["masked_lm_weights"]
            label = features["labels"] 

        tensor_embedding_table = tf.get_variable("embedding_table",
                                                 shape=[config.vocab_size,config.vocab_vec_size],
                                                 trainable=embedding_table_trainable)

        def init_embedding_table(scaffold,sess):
            sess.run(tensor_embedding_table.initializer, {tensor_embedding_table.initial_value: embedding_table})

        scaffold = tf.train.Scaffold(init_fn=init_embedding_table)

        model = modeling.SimTransformer(
            config=config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            embedding_table=tensor_embedding_table,
            embedding_table_trainable=embedding_table_trainable)

        output_transformer = model.get_sequence_output()
        pooled_output_transformer = model.get_pooled_output()

        num_heads = config.num_attention_heads
        # (pred_label_loss, pred_label_example_loss, pred_label_probs) = get_pred_label_output(
            # config, pooled_output_transformer, label, is_training)

        
        if mode != tf.estimator.ModeKeys.PREDICT:
            (pred_label_loss, pred_label_example_loss, pred_label_probs) = get_pred_label_output(
                config, pooled_output_transformer, label, is_training)
            (masked_lm_loss, masked_lm_example_loss) = get_masked_lm_output(
                config, output_transformer, model.get_embedding_table(),masked_lm_positions,
                masked_lm_ids, masked_lm_weights,num_heads,rng)

            total_loss = pred_label_loss + masked_lm_loss
            tf.summary.scalar("pred_label_loss",pred_label_loss)
            tf.summary.scalar("masked_lm_loss",masked_lm_loss)
        else:
            pred_label_probs = get_pred_label_output(
                config, pooled_output_transformer, None, is_training)

            
        tvars = tf.trainable_variables()
        initialized_variable_names = {}

        if init_checkpoint:
            (assignment_map, initialized_variable_names
            ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ",*INIT_FROM_CKPT*"
            tf.logging.info(" name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        # scaffold = None
        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps)

            logging_hook = tf.train.LoggingTensorHook({"total_loss":total_loss,"label_loss":pred_label_loss,"lm_loss":masked_lm_loss}, every_n_iter=1)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                training_hooks=[logging_hook],
                scaffold=scaffold)

        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(loss):
                return {
                    "eval_loss":loss
                }
            eval_metrics = (metric_fn,[])
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=pred_label_loss,
                eval_metrics=eval_metrics)
        else:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                scaffold=scaffold,
                predictions=pred_label_probs)
        return output_spec

    return model_fn
            
def get_pred_label_output(config,input_tensor,label=None,is_training=False):
    with tf.variable_scope("pred_label"):
        output_weights = tf.get_variable(
            "output_weights",
            shape=[config.label_size, config.hidden_size],
            initializer=modeling.create_initializer(config.initializer_range))
        output_bias = tf.get_variable(
            "output_bias",
            shape=[config.label_size],
            initializer=tf.zeros_initializer())

        logits = tf.matmul(input_tensor, output_weights, transpose_b=True) 
        logits = tf.nn.bias_add(logits, output_bias)
        label_probs = tf.nn.softmax(logits, axis=-1)
        # if is_training:
        if is_training:
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            label = tf.reshape(label, [-1])
            one_hot_label = tf.one_hot(label, depth=2, dtype=tf.float32)
            per_example_loss = -tf.reduce_sum(one_hot_label * log_probs, axis = -1)
            loss = tf.reduce_mean(per_example_loss)
            return (loss,per_example_loss,label_probs)
        else:
            return label_probs

def build_negative_sample_weights(input_tensor,label_ids,weights,sample_scope,neg_num,rng,num_heads=None):
    input_shape = modeling.get_shape_list(input_tensor)
    tmp_batch_size = input_shape[0]
    # neg_indexs = rng.randint(sample_scope,size=(tmp_batch_size,neg_num))
    neg_sample_ids = tf.random.uniform(shape=[tmp_batch_size,neg_num],
                                   minval=0,
                                   maxval=sample_scope-1,
                                   dtype=tf.int64)
    all_ids = tf.concat([label_ids,neg_sample_ids],axis=1)
    # logits_weights, shape: [batch_size, window_size, hidden_size]
    logits_weights = tf.gather(weights, all_ids)
    if num_heads:
        logits_weights = tf.tile(logits_weights,[1,1,num_heads])
    
    return logits_weights

        
def get_masked_lm_output(config, input_tensor, output_weights, positions,
                        label_ids, label_weights,num_heads,rng):
    """
    Args:
        output_weights: word embedding table, shape: [vocab_size,hidden_size]
        positions: positions of lm prediction 
        label_ids: lm predicted word id, shape: [batch_size,position_size]
        label_weights: because lm predictions size is not all same, we need
            to padding 0 for some example, if 0 padding, label_weights is 0.0
    """
    batch_size = tf.shape(input_tensor)[0]
    position_size = tf.shape(label_weights)[1]

    # make input_tensor to shape [batch_size*position_size,hidden_size]
    input_tensor = gather_indexes(input_tensor, positions)
    with tf.variable_scope("lm_predictions"):
        with tf.variable_scope("lm_out_layer"):
            input_tensor = tf.layers.dense(
                input_tensor,
                units=config.hidden_size,
                activation=modeling.get_activation(config.hidden_act),
                kernel_initializer=modeling.create_initializer(
                    config.initializer_range))
            input_tensor = modeling.layer_norm(input_tensor)

        output_bias = tf.get_variable(
            "lm_output_bias",
            shape=[config.vocab_size],
            initializer=tf.zeros_initializer())

        neg_sample_num = config.neg_sample_num 
        # reshape label_ids from [batch_size,position_size] 
        # to [batch_size*position_size]
        label_ids = tf.reshape(label_ids,[-1,1])
        # label_weights = tf.reshape(label_weights, [-1])

        # negative shape [batch_size*position_size, 1 + neg_sample_num, hidden_size]
        negative_sample_weights = build_negative_sample_weights(input_tensor,label_ids,output_weights,config.vocab_size,neg_sample_num,rng,num_heads)
        negative_sample_bias = build_negative_sample_weights(input_tensor,label_ids,output_bias,config.vocab_size,neg_sample_num,rng)

        # change input_tensor shape to: [batch_size*position_size,1,hidden_size], 
        # in order to get logits as shape [batch_size*position_size, 1, 1 + neg_sample_num]
        input_tensor = tf.expand_dims(input_tensor,axis=1)

        # reshape negative_sample_weights, res: [batch_size*position_size, hidden_size, 1 + neg_sample_num]
        negative_sample_weights = tf.transpose(negative_sample_weights,perm=[0,2,1])

        # input_tensor:[batch_size*position_size , 1, hidden_size]
        # negative_sample_weights:[batch_size*position_size, hidden_size, 1 + neg_sample_num]
        # logits: [batch_size, position_size, 1 + neg_sample_num]
        logits = tf.matmul(input_tensor, negative_sample_weights)
        # reshape to [batch_size*position_size, 1 + neg_sample_ids]
        logits = tf.reshape(logits, [-1,1+neg_sample_num])  
        tf.logging.debug("logits shape:%s",str(logits.shape.as_list()))
        tf.logging.debug("bias shape:%s",str(negative_sample_bias.shape.as_list()))
        logits = tf.add(logits, negative_sample_bias)

        neg_label = tf.zeros(
            shape=[batch_size*position_size,neg_sample_num],
            dtype=tf.float32,
            name="neg_label")
        pos_label = tf.ones(
            shape=[batch_size*position_size,1],
            dtype=tf.float32,
            name="pos_label")
        one_hot_labels = tf.concat([pos_label,neg_label],axis=1)

        #per_example_loss shape: [bactch_size*position_size,1 + neg_sample_ids]
        per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=one_hot_labels,
                                                                   logits=logits,
                                                                   name="pred_lm_loss")

        tf.logging.debug("per_example_loss shape:%s"%(per_example_loss.shape.as_list()))

        #per_example_loss shape: [batch_size,position_size]
        per_example_loss = tf.reduce_sum(tf.reshape(per_example_loss,shape=[batch_size,position_size,1 + neg_sample_num]),
                                         axis=[-1])
        # The `positions` tensor might be zero-padded (if the sequence is too
        # short to have the maximum number of predictions). The `label_weights`
        # tensor has a value of 1.0 for every real predictions and 0.0 for the
        # padding predictions.
        # per_example_loss shape: [batch_size*positions,1 + neg_sample_num]

        tf.logging.debug("per_example_loss size:%s"%(per_example_loss.shape.as_list()))
        tf.logging.debug("label_weights size:%s"%(label_weights.shape.as_list()))
        per_example_loss = tf.reduce_sum(per_example_loss * label_weights)
        numerator = tf.reduce_sum(per_example_loss)
        denominator = tf.reduce_sum(label_weights) + 1e-5
        loss = numerator / denominator

    return (loss, per_example_loss)

def gather_indexes(sequence_tensor,positions):
    """Gathers the vectors at the specific positions over a minibatch.""" 
    sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    hidden_size = sequence_shape[2]

    flat_offsets = tf.reshape(
        tf.range(0,batch_size, dtype=tf.int64) * seq_length, [-1,1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                      [batch_size * seq_length, hidden_size])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor

def input_fn_builder():
    def input_fn(params):
        pass


def file_based_input_fn_builder_train(input_file,
                                      seq_length,
                                      max_lm_pred,
                                      label_size,
                                      is_training,
                                      drop_remainder):
    """
    """
    name_to_features = {
        "input_ids":tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask":tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids":tf.FixedLenFeature([seq_length], tf.int64),
        "masked_lm_ids":tf.FixedLenFeature([max_lm_pred], tf.int64),
        "masked_lm_positions":tf.FixedLenFeature([max_lm_pred], tf.int64),
        "masked_lm_weights":tf.FixedLenFeature([max_lm_pred],tf.float32),
        "labels":tf.FixedLenFeature([1], tf.int64),
    }

    # def _decode_record(record, name_to_features):
        # """Decodes a record to a Tensorflow example."""
        # example = tf.parse_single_example(record, name_to_features)
        # for name in list(example.keys()):
            # t = example[name]
            # if t.dtype == tf.int64:
                # t = tf.to_int32(t)
            # example[name] = t
        # return example

    def input_fn(params):
        batch_size = params["batch_size"]
        # epoch = params["epoch"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                # lambda record: _decode_record(record, name_to_features),
                lambda record:tf.parse_single_example(record,name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))
        return d
    return input_fn

def file_based_input_fn_builder_predict(input_file,
                                        seq_length,
                                        vocab_file,
                                        vocab_vec_file):
    from segmentor import segmentor
    segmenter = segmentor.Segmentor()
    tokenizer = tokenization.Tokenizer(vocab_file,vocab_vec_file)
    # name_to_features = {
        # "input_ids":tf.FixedLenFeature([seq_length], tf.int64),
        # "input_mask":tf.FixedLenFeature([seq_length], tf.int64),
        # "segment_ids":tf.FixedLenFeature([seq_length], tf.int64),
        # "masked_lm_ids":tf.FixedLenFeature([max_lm_pred], tf.int64),
        # "masked_lm_positions":tf.FixedLenFeature([max_lm_pred], tf.int64),
        # "masked_lm_weights":tf.FixedLenFeature([max_lm_pred],tf.float32),
        # "labels":tf.FixedLenFeature([1], tf.int64),
    # }
    def generate_fn():
        with open(input_file,"r") as fp:
            for line in fp:
                items = line.strip().split('\t')
                query = items[0]
                doc = items[1]

                tokens_a = segmenter.segment(query)
                tokens_b = segmenter.segment(doc)
                # tokens_a = items[0].split(',')
                # tokens_b = items[1].split(',')
                
                tokens = [CodeUtil.UTFNormalize("[CLS]")] + tokens_a + [CodeUtil.UTFNormalize("[SEP]")] + tokens_b
                n_a = len(tokens_a)
                n_b = len(tokens_b)
                input_mask = [0]*seq_length
                segment_ids = [0] * seq_length

                if (n_a + 2 + n_b) > seq_length:
                    n_b = seq_length - n_a - 2
                     
                for i in range(n_a + 2 + n_b):
                    input_mask[i] = 1

                for i in range(n_a + 2):
                    segment_ids[i] = 1
                for i in range(n_a+2, n_a+2+n_b):
                    segment_ids[i] = 0

                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                if len(input_ids) > seq_length:
                    input_ids = input_ids[0:seq_length]
                assert len(input_ids) <= seq_length
                while len(input_ids) < seq_length:
                    input_ids.append(0)

                assert len(input_ids) == seq_length
                assert len(input_mask) == seq_length
                assert len(segment_ids) == seq_length
                # features = collections.OrderedDict()
                # features["input_ids"] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(input_ids)))
                # features["input_mask"] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(input_mask)))
                # features["segment_ids"] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(segment_ids)))
                # input_ids_tensor = tf.train.Feature(int64_list=tf.train.Int64List(value=list(input_ids)))
                # input_mask_tensor = tf.train.Feature(int64_list=tf.train.Int64List(value=list(input_mask)))
                # segment_ids_tensor = tf.train.Feature(int64_list=tf.train.Int64List(value=list(segment_ids)))
                # yield (input_ids_tensor,input_mask_tensor,segment_ids_tensor)
                yield (input_ids,input_mask,segment_ids)

    def input_fn(params):
        dataset = tf.data.Dataset.from_generator(generate_fn,
                                                 output_types=(tf.int64,
                                                               tf.int64,
                                                               tf.int64),
                                                 output_shapes=(tf.TensorShape([seq_length]),
                                                               tf.TensorShape([seq_length]),
                                                               tf.TensorShape([seq_length]))
                                                )

        input_features = dataset.batch(params["batch_size"])
        iter = input_features.make_one_shot_iterator()
        iterdata = iter.get_next()
        # input_ids_tensor = tf.train.Feature(int64_list=tf.train.Int64List(value=list(iterdata[0])))
        # input_mask_tensor = tf.train.Feature(int64_list=tf.train.Int64List(value=list(iterdata[1])))
        # segment_ids_tensor = tf.train.Feature(int64_list=tf.train.Int64List(value=list(iterdata[2])))
        input_ids_tensor = iterdata[0]
        input_mask_tensor = iterdata[1]
        segment_ids_tensor = iterdata[2]
        feat_dict = {
            "input_ids":input_ids_tensor,
            "input_mask":input_mask_tensor,
            "segment_ids":segment_ids_tensor,
        }
        return feat_dict,None
    return input_fn

def loadEmbeddingTable(embedding_table_file):
    embedding_table = tokenization.load_vocab_vec_file(embedding_table_file)
    return embedding_table

def main():
    tf.logging.set_verbosity(tf.logging.DEBUG)
    config = modeling.Config(vocab_size=FLAGS.vocab_size,
                            vocab_vec_size=FLAGS.vocab_vec_size,
                            hidden_size=6*200,
                            num_hidden_layers=3,
                            num_attention_heads=6,
                            intermediate_size=516,
                            hidden_act="gelu",
                            hidden_dropout_prob=0.1,
                            attention_probs_dropout_prob=0.1,
                            max_position_embeddings=128,
                            type_vocab_size=2,
                            label_size=2,
                            initializer_range=0.02,
                            neg_sample_num=FLAGS.neg_sample_num)

    tf.gfile.MakeDirs(FLAGS.output_dir)
    # input_files = []
    input_file = FLAGS.input_file
    tf.logging.info("*** Input File ***")
    tf.logging.info(" %s"%input_file)

    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.output_dir,
        tf_random_seed=None,
        save_summary_steps=1,
        save_checkpoints_steps=5,
        keep_checkpoint_max=2,
        )

    embedding_table = None
    if FLAGS.embedding_table_file:
        embedding_table = loadEmbeddingTable(FLAGS.embedding_table_file)

    rng = random.Random(FLAGS.random_seed)
    model_fn = model_fn_builder(
        config=config,
        rng=rng,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=FLAGS.num_train_steps,
        num_warmup_steps=FLAGS.num_warmup_steps,
        embedding_table=embedding_table,
        embedding_table_trainable=FLAGS.embedding_table_trainable)

    params = dict()
    params['batch_size'] = FLAGS.train_batch_size
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params=params)


    if FLAGS.do_train:
        tf.logging.info("***** Running training *****")
        tf.logging.info(" Batch size = %d", FLAGS.train_batch_size)
        train_input_fn = file_based_input_fn_builder_train(
            input_file=input_file,
            seq_length=FLAGS.max_seq_length,
            label_size=2,
            max_lm_pred=FLAGS.max_predictions_per_seq,
            is_training=True,
            drop_remainder=False)
        estimator.train(input_fn=train_input_fn,max_steps=FLAGS.num_train_steps)
    elif FLAGS.do_predict:
        tf.logging.info("***** Running Predict *****")
        predict_input_fn = file_based_input_fn_builder_predict(
            input_file=input_file,
            seq_length=FLAGS.max_seq_length,
            vocab_file=FLAGS.vocab_file,
            vocab_vec_file=FLAGS.vocab_vec_file)
        pred_res = estimator.predict(input_fn=predict_input_fn,
                          predict_keys=None,
                          checkpoint_path=None,
                          yield_single_examples=True)

        wfp = open("pred_res","w")
        for res in pred_res:
            print res
            str_res = [str(x) for x in res]
            wfp.write("\t".join(str_res)+'\n')
        wfp.close()

if __name__ == "__main__":
    main()

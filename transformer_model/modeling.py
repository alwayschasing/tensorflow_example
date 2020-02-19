#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
"""
import copy
import math
import tensorflow as tf
import six
import collections
import re

class Config(object):
    def __init__(self,
                 vocab_size,
                 vocab_vec_size=200,
                 hidden_size=6*128,
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
                 neg_sample_num=10):
        self.vocab_size = vocab_size
        self.vocab_vec_size = vocab_vec_size
        self.hidden_size=hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads=num_attention_heads
        self.intermediate_size=intermediate_size
        self.hidden_act= hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings=max_position_embeddings
        self.type_vocab_size=type_vocab_size
        self.label_size=label_size
        self.initializer_range=initializer_range
        self.neg_sample_num=neg_sample_num

class SimTransformer(object):
    """query similarity model based on Transformer
    """
    def __init__(self,
                 config,
                 is_training,
                 input_ids,
                 embedding_table=None,
                 embedding_table_trainable=False,
                 input_mask=None,
                 token_type_ids=None,
                 scope=None):
        """Constructor for SimTransformer
        Args:
            input_ids:
            embedding_table:
        """
        config = copy.deepcopy(config)
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        input_shape = get_shape_list(input_ids, expected_rank=2)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        tf.logging.debug("sequence_output shape:%s"%(str(input_shape)))

        if token_type_ids is None:
            token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

        with tf.variable_scope(scope, default_name="SimTransformer"):
            with tf.variable_scope("embeddings"):

                # self.embedding_table = tf.get_variable(
                    # name="embedding_table",
                    # shape=[config.vocab_size,config.vocab_vec_size],
                    # dtype=tf.float32,
                    # initializer=tf.constant_initializer(embedding_table),
                    # trainable=embedding_table_trainable)
                # self.embedding_table = tf.get_variable(
                    # dtype=tf.float32,
                    # # shape=[config.vocab_size,config.vocab_vec_size],
                    # initializer=tf.ones(shape=(config.vocab_size,config.vocab_vec_size)),
                    # name="embedding_table")
                # self.embedding_table.load(embedding_table)
                self.embedding_table = embedding_table
                # sess = tf.get_default_session()
                # sess.run(self.embedding_table,feed_dict={self.embedding_table:embedding_table})
                # Perform embedding lookup on the word ids
                # embedding_output = embedding_lookup(embedding_table,input_ids)                
                self.embedding_output = tf.nn.embedding_lookup(embedding_table,input_ids)
                # Add positional embeddings and token type embeddings, then layer
                # normalize and perform dropout
                self.embedding_output = embedding_postprocessor(
                    input_tensor=self.embedding_output,
                    num_heads=config.num_attention_heads,
                    use_token_type=True,
                    token_type_ids=token_type_ids,
                    token_type_vocab_size=config.type_vocab_size,
                    token_type_embedding_name="token_type_embeddings",
                    use_position_embeddings=True,
                    position_embedding_name="position_embeddings",
                    initializer_range=config.initializer_range,
                    max_position_embeddings=config.max_position_embeddings,
                    dropout_prob=config.hidden_dropout_prob)

            with tf.variable_scope("transformer"):
                # This converts a 2D mask of shape [batch_size, seq_length] to
                # a 3D mask of shape [batch_size, seq_length, seq_length] which
                # is used for the attention scores.
                attention_mask = create_attention_mask_from_input_mask(
                    input_ids, input_mask)

                # Run the stacked transformer
                # `seqence_output` shape = [batch_size, seq_length, hidden_size].
                # self.all_encoder_layers = transformer_model(
                self.sequence_output = transformer_model(
                    input_tensor=self.embedding_output,
                    attention_mask=attention_mask,
                    hidden_size=config.hidden_size,
                    num_hidden_layers=config.num_hidden_layers,
                    num_attention_heads=config.num_attention_heads,
                    intermediate_size=config.intermediate_size,
                    intermediate_act_fn=get_activation(config.hidden_act),
                    hidden_dropout_prob=config.hidden_dropout_prob,
                    attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                    initializer_range=config.initializer_range,
                    do_return_all_layers=False)
            # all_encoder_layers_shape = get_shape_list(self.all_encoder_layers)
            # tf.logging.debug("all_encoder_layers_shape:%s"%(str(all_encoder_layers_shape)))
            # self.sequence_output = self.all_encoder_layers[-1]
            check_shape = get_shape_list(self.sequence_output)
            tf.logging.debug("sequence_output shape:%s"%(str(check_shape)))

            with tf.variable_scope("pooler"):
                first_token_tensor = tf.squeeze(self.sequence_output[:,0:1,:],axis=1)
                self.pooled_output = tf.layers.dense(
                    first_token_tensor,
                    config.hidden_size,
                    activation=tf.tanh,
                    kernel_initializer=create_initializer(config.initializer_range))

    def get_sequence_output(self):
        return self.sequence_output

    def get_pooled_output(self):
        return self.pooled_output

    def get_embedding_table(self):
        return self.embedding_table

def gelu(input_tensor):
    """Gaussian Error Linear Unit.
    This is a smoother version of the RELU.
    Original paper:https://arxiv.org/abs/1606.08415
    
    Args:
        input_tensor: float Tensor to perform activation.

    Returns:
        `input_tensor` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
    return input_tensor * cdf


def get_activation(activation_string):
    """
    Returns:
        A python function corresponding to the activation function. If
        `activation_string` is None, empty, or "linear", this will return None.
        If `activation_string` is not a string.
    Raises:
        ValueError: The `activation_string` does not correspond to a known
        activation.
    """
    if not activation_string:
        return None
    act = activation_string.lower()
    if act == "linear":
        return None
    elif act == "relu":
        return tf.nn.relu
    elif act == "gelu":
        return gelu
    elif act == "tanh":
        return tf.tanh
    else:
        raise ValueError("Unsupported activation:%s"%act)

def assert_rank(tensor, expected_rank, name=None):
    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape=%s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))

def get_shape_list(tensor, expected_rank=None, name=None):
    """
    Returns a list of the shape of tensor, preferring static dimensions.
    Args:
        tensor: A tf.Tensor object to find the shape of.
        expected_rank: (optional) int. The expected rank of `tensor`. If this is
        specified and the `tensor` has a different rank, and exception will be
        thrown.
        name: Optional name of the tensor for the error message.
    Returns:
        A list of dimensions of the shape of tensor. All static dimensions will
        be returned as python integers, and dynamic dimensions will be returned
        as tf.Tensor scalars.
    """
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    non_static_indexes = []
    shape = tensor.shape.as_list()
    for (index,dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor,out_type=tf.int64)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape

def create_initializer(initializer_range=0.02):
    """Create a `truncated_normal_initializer` with the given range"""
    return tf.truncated_normal_initializer(stddev=initializer_range)

def layer_norm(input_tensor, name=None):
    return tf.contrib.layers.layer_norm(
        inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1,scope=name)

def dropout(input_tensor, dropout_prob):
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor
    output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
    return output 

def layer_norm_and_dropout(input_tensor, dropout_prob, name=None):
    output_tensor = layer_norm(input_tensor, name)
    output_tensor = dropout(output_tensor, dropout_prob)
    return output_tensor

def embedding_lookup(embedding_table, input_ids):
    """
    Args:
        input_ids: int32 Tensor of shape [batch_size, seq_length], containing word ids.
    Returns:
        float Tensor of shape [batch_size, seq_length, embedding_size].
    """
    # if input_ids.shape.ndims == 2:
        # input_ids = tf.expand_dims(input_ids, axis=[-1])
    output = tf.nn.embedding_lookup(embedding_table, input_ids)
    return output

def embedding_postprocessor(input_tensor,
                            num_heads,
                            use_token_type=False,
                            token_type_ids=None,
                            token_type_vocab_size=2,
                            token_type_embedding_name="token_type_embeddings",
                            use_position_embeddings=True,
                            position_embedding_name="position_embeddings",
                            initializer_range=0.02,
                            max_position_embeddings=512,
                            dropout_prob=0.1):
    """Perform various post-processing on a word embedding tensor.
    Args:
        input_tensor: float Tensor of shape [batch_size, seq_length, embedding_size].
        num_heads: int. num of heads
        use_token_type: bool. Whether to add embeddings for `token_type_ids`.
        token_type_ids:(optional) int32 Tensor of shape [batch_size, seq_length]
        max_position_embeddings: int. Maximum sequence length that might ever be used 
            with this model.
    Returns:
        float tensor with shape [batch_size, seq_length, embedding_size*num_heads].
    """
    input_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    width = input_shape[2]

    if seq_length > max_position_embeddings:
        raise ValueError("The seq length (%d) cannot be greater than "
                         "`max_position_embeddings` (%d)" % 
                         (seq_length, max_position_embeddings))

    output = input_tensor
    # output = tf.tile(output,[1,1,num_heads])
    if use_token_type:
        if token_type_ids is None:
            raise ValueError("`token_type_ids` must be specified if "
                             "`use_token_type` is True.")

        token_type_table = tf.get_variable(
            name=token_type_embedding_name,
            shape=[token_type_vocab_size, width],
            initializer=create_initializer(initializer_range))
        flat_token_type_ids = tf.reshape(token_type_ids,[-1])
        one_hot_ids = tf.one_hot(flat_token_type_ids, depth=token_type_vocab_size)
        token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
        token_type_embeddings = tf.reshape(token_type_embeddings,
                                           [batch_size, seq_length, width])
        # token_type_embeddings = tf.tile(output,[1,1,num_heads])
        output = tf.math.add(output,token_type_embeddings)

    if use_position_embeddings:
        full_position_embeddings = tf.get_variable(
            name=position_embedding_name,
            shape=[max_position_embeddings, width],
            initializer=create_initializer(initializer_range))

        # Since the position embedding table is a learned variable, we create it 
        # using a (long) sequence length `max_position_embeddings`. The actual
        # sequence length might be shorter than this, for faster training of 
        # tasks that do not have long sequences.

        # `full_position_embeddings` is effectively an embedding table for
        # for position [0,1,2,..., max_position_embeddings-1], and the current
        # sequence has positions [0,1,2,...,seq_length-1], so we can just 
        # perform a slice.
        if seq_length < max_position_embeddings:
            position_embeddings = tf.slice(full_position_embeddings, [0,0],
                                          [seq_length,-1])
        else:
            position_embeddings = full_position_embeddings
        
        num_dims = len(output.shape.as_list())
        
        # Only the last two dimensions are relevant (`seq_length` and `width`), so
        # we broadcast among the first dimensions, which is typically just 
        # the batch size.
        position_broadcast_shape = []
        for _ in range(num_dims - 2):
            position_broadcast_shape.append(1)
        
        position_broadcast_shape.extend([seq_length,width])
        position_embeddings = tf.reshape(position_embeddings,
                                         position_broadcast_shape)
        # position_embeddings = tf.tile(position_embeddings,[1,1,num_heads])
        output = tf.math.add(output,position_embeddings)

    output = tf.tile(output,[1,1,num_heads])
    output = layer_norm_and_dropout(output, dropout_prob)
    return output

def create_attention_mask_from_input_mask(from_tensor, to_mask):
    """Create 3D attention mask from a 2D tensor mask.
    Args:
        from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
        to_mask: int32 Tensor of shape [batch_size, to_seq_lenght].
    """
    from_shape = get_shape_list(from_tensor, expected_rank=[2,3])
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]

    to_shape = get_shape_list(to_mask, expected_rank=2)
    to_seq_length = to_shape[1]

    to_mask = tf.cast(
        tf.reshape(to_mask, [batch_size,1,to_seq_length]), tf.float32)
    
    broadcast_ones = tf.ones(
        shape=[batch_size, from_seq_length, 1], dtype=tf.float32)
    mask = broadcast_ones * to_mask
    return mask

def attention_layer(from_tensor,
                    to_tensor,
                    attention_mask=None,
                    num_attention_heads=1,
                    size_per_head=128,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    do_return_2d_tensor=False,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None):
    """
    """
    def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                             seq_length, width):
        output_tensor = tf.reshape(
            input_tensor, [batch_size, seq_length, num_attention_heads, width])

        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

    from_shape = get_shape_list(from_tensor, expected_rank=[2,3])
    to_shape = get_shape_list(to_tensor, expected_rank=[2,3])
    if len(from_shape) != len(to_shape):
        raise ValueError(
            "The rank of `from_tensor` must match the rank of `to_tensor`.")

    if len(from_shape) == 3:
        batch_size = from_shape[0]
        from_seq_length = from_shape[1]
        to_seq_length = to_shape[1]
    elif len(from_shape) == 2:
        if (batch_size is None or from_seq_length is None or to_seq_length is None):
            raise ValueError(
                "When passing in rank 2 tensors to attention_layer, the values"
                " for `batch_size`, `from_seq_length`, and `to_seq_length` "
                " must all be specified.")

    # Scalar dimensions referenced here:
    # B = batch size (number of seqences)
    # F = `from_tensor` seqence length
    # T = `to_tensor` seqence length
    # N = `num_attention_heads`
    # H = `size_per_head`

    from_tensor_2d = reshape_to_matrix(from_tensor)
    to_tensor_2d = reshape_to_matrix(to_tensor)

    # `query_layer` = [B*F, N*H]
    query_layer = tf.layers.dense(
        from_tensor_2d,
        num_attention_heads * size_per_head,
        activation=query_act,
        name="query",
        kernel_initializer=create_initializer(initializer_range))

    # `key_layer` = [B*T, N*H]
    key_layer = tf.layers.dense(
        to_tensor_2d,
        num_attention_heads * size_per_head,
        activation=key_act,
        kernel_initializer=create_initializer(initializer_range))

    # `value_layer` = [B*T, N*H]
    value_layer = tf.layers.dense(
        to_tensor_2d,
        num_attention_heads * size_per_head,
        activation=value_act,
        name="value",
        kernel_initializer=create_initializer(initializer_range))

    # `query_layer` = [B, N, F, H]
    query_layer = transpose_for_scores(
        query_layer, batch_size, num_attention_heads,
        from_seq_length, size_per_head)

    # `key_layer` = [B, N, T, H]
    key_layer = transpose_for_scores(key_layer, batch_size,
                                     num_attention_heads, to_seq_length,
                                     size_per_head)

    # Take the dot product between "query" and "key" to get the raw
    # attention_scores.
    # `attention_scores` = [B, N, F, T]
    attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
    attention_scores = tf.multiply(attention_scores,
                                   1.0/math.sqrt(float(size_per_head)))

    if attention_mask is not None:
        # `attention_mask` = [B, 1, F, T]
        attention_mask = tf.expand_dims(attention_mask, axis=[1])

        adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

        attention_scores += adder
    
    attention_probs = tf.nn.softmax(attention_scores)
    attention_probs = dropout(attention_probs, attention_probs_dropout_prob)

    # `value_layer` = [B, T, N, H]
    value_layer = tf.reshape(
        value_layer,
        [batch_size, to_seq_length, num_attention_heads, size_per_head])

    # `value_layer` = [B, N ,T ,H] 
    value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

    # `context_layer` = [B, N, F, H]
    context_layer = tf.matmul(attention_probs, value_layer)

    # `context_layer` = [B, F, N, H]
    context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

    if do_return_2d_tensor:
        # `context_layer` = [B*F, N*V]
        context_layer = tf.reshape(
            context_layer,
            [batch_size * from_seq_length, num_attention_heads * size_per_head])
    else:
        # `context_layer` = [B, F, N*V]
        context_layer = tf.reshape(
            context_layer,
            [batch_size , from_seq_length, num_attention_heads*size_per_head])

    return context_layer

def transformer_model(input_tensor,
                      attention_mask=None,
                      hidden_size=768,
                      num_hidden_layers=3,
                      num_attention_heads=6,
                      intermediate_size=516,
                      intermediate_act_fn=gelu,
                      hidden_dropout_prob=0.1,
                      attention_probs_dropout_prob=0.1,
                      initializer_range=0.02,
                      do_return_all_layers=False):
    """
    Args:
        input_tensor: float Tensor of shape[batch_size, seq_length, hidden_size].
        attention_mask: (optional) int32 Tensor of shape [batch_size, seq_lenght, 
        seq_length], with 1 for positions that can be attended to and 0 in 
        positions that should not be.
    """

    if hidden_size % num_attention_heads != 0:
        raise ValueError(
            "The hidden size (%d) is not a multiple of the number of attention"
            " heads (%d)" % (hidden_size, num_attention_heads))

    attention_head_size = int(hidden_size / num_attention_heads)
    input_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    input_width = input_shape[2]
    
    if input_width != hidden_size:
        raise ValueError("The width of the input tensor (%d) != hidden size (%d)" % (input_width, hidden_size))
    tf.logging.debug("batch_size:%s,seq_length:%s,input_width:%s"%(str(batch_size),str(seq_length),str(input_width)))

    prev_output = reshape_to_matrix(input_tensor)

    all_layer_outputs = []
    for layer_idx in range(num_hidden_layers):
        with tf.variable_scope("layer_%d" % layer_idx):
            layer_input = prev_output

            with tf.variable_scope("attention"):
                with tf.variable_scope("self"):
                    attention_head = attention_layer(
                        from_tensor=layer_input,
                        to_tensor=layer_input,
                        attention_mask=attention_mask,
                        num_attention_heads=num_attention_heads,
                        size_per_head=attention_head_size,
                        attention_probs_dropout_prob=attention_probs_dropout_prob,
                        initializer_range=initializer_range,
                        do_return_2d_tensor=True,
                        batch_size=batch_size,
                        from_seq_length=seq_length,
                        to_seq_length=seq_length)
            
                attention_output = attention_head
                with tf.variable_scope("output"):
                    attention_output = tf.layers.dense(
                        attention_output,
                        hidden_size,
                        kernel_initializer=create_initializer(initializer_range))
                    attention_output = dropout(attention_output, hidden_dropout_prob)
                    attention_output = layer_norm(attention_output + layer_input)

            with tf.variable_scope("intermediate"):
                intermediate_output = tf.layers.dense(
                    attention_output,
                    intermediate_size,
                    activation=intermediate_act_fn,
                    kernel_initializer=create_initializer(initializer_range))
            
            with tf.variable_scope("output"):
                layer_output = tf.layers.dense(
                    intermediate_output,
                    hidden_size,
                    kernel_initializer=create_initializer(initializer_range))
                layer_output = dropout(layer_output, hidden_dropout_prob)
                layer_output = layer_norm(layer_output + attention_output)
                prev_output = layer_output
                all_layer_outputs.append(layer_output)

    if do_return_all_layers:
        final_outputs = []
        for layer_output in all_layer_outputs:
            final_output = reshape_from_matrix(layer_output, input_shape)
            final_outputs.append(final_output)
        return final_outputs
    else:
        final_output = reshape_from_matrix(prev_output, input_shape)
        return final_output

def reshape_from_matrix(output_tensor, orig_shape_list):
    if len(orig_shape_list) == 2:
        return output_tensor
    output_shape = get_shape_list(output_tensor)
    orig_dims = orig_shape_list[0:-1]
    width = output_shape[-1]
    return tf.reshape(output_tensor, orig_dims + [width])

def reshape_to_matrix(input_tensor):
    ndims = input_tensor.shape.ndims
    if ndims < 2:
        raise ValueError("Input tensor must have at least rank 2. Shape = %s"
                         % (input_tensor.shape))
    if ndims == 2:
        return input_tensor

    width = input_tensor.shape[-1]
    output_tensor = tf.reshape(input_tensor, [-1, width])
    return output_tensor

def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$",name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)
    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1
    return (assignment_map, initialized_variable_names)

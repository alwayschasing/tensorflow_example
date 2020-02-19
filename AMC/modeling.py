#!/usr/bin/env python
# -*- coding:utf-8 -*-
import tensorflow as tf

class AMT(object):
    """AMT model(Attention guided Multi-model Correlation Learning for Image Search).

    MTN(Multi-model inter-attention network)
    VAN(Visual intra-attention network)
    LAN(Language intra-attention network)
    """
    def __init__(self,
                config,
                is_training,
                input_q,
                input_K,
                input_v,
                scope=None):
        """Constructor for AMT model
        Args:
            
        """
        self.q = input_q  #query feature vector
        self.K = input_K  #txt feature vector of image
        self.v = input_v  #image feature vector 
        self.is_training = is_training
        with tf.variable_scope(scope, default_name="AMT"):
            if config["use_VAN"]:
                self.v_q = self.build_VAN(config)
            else:
                self.v_q = None

            if config["use_LAN"]:
                self.k_q = self.build_LAN(config)
            else:
                self.k_q = None

            self.q_m,self.x_q = self.build_MTN(config)
            if self.is_training:
                with tf.variable_scope("loss"):
                    """
                    argmin L
                    L = max(0, a - <q_m,x_q+> + <q_m,x_q->), x_q+:positive instance,
                    x_q[0]:positive
                    x_q[1:]:negative
                    """
                    alpha = tf.place_holder(shpae=[1,],name="margin")
                    cosin_dist = get_cosine_distance(self.q_m,self.x_q);
                    loss = 0 
                    n_negative = config["n_negative"]
                    n_instance = tf.shape(self.q)[0]/(n_negative+1) 
                    loss = tf.substract(alpha,cosin_dist[0*(n_negative+1)])
                    for tp in range(n_negative):
                        loss = tf.add(loss,cosin_dist[0*(n_negative+1)+tp+1])
                        loss = tf.maximum(0,loss) 
                    for i in range(1,n_instance):
                        accumulate = tf.substract(alpha,cosin_dist[i*(n_negative+1)])
                        for j in range(n_negative):
                            accumulate = tf.add(accumulate,cosin_dist[i*(n_negative+1)+j+1])
                        accumulate = tf.maximum(0,accumulate)
                        loss = tf.add(loss,accumulate)
                    self.loss = loss
                

    def load_config(self,config):
        """
        util function
        config: {"n_negative":0,
                }
        """
        self.n_negative = config["n_negative"]
    def build_VAN(self,config):
        with tf.variable_scope("VAN"):
            q_shape = tf.shape(self.q)
            v_shape = tf.shape(self.v)
            dq = q_shape[1]
            dv = v_shape[1]
            r = config["r"] # 
            d = config["d"] # 
            W_qs = tf.get_variable(
                name = "W_qs",
                shape = [r,r,d,dq],
                dtype = tf.float32,
                initializer = create_initializer(0.02))
            b_qs = tf.get_variable(
                name = "b_qs",
                shape = [r,r,d],
                dtype = tf.float32,
                initializer = tf.zeros(shape=[],dtype=tf.float32)
                )
            #[r,r,d,dq]*[dq,1] = [r,r,d]
            s_q = tf.nn.relu(tf.add(tf.matmul(W_qs,tf.transpose(self.q)),b_qs))
            W_v = tf.get_variable(
                name = "W_v",
                shape = [1,1,dv,d],
                dtype = tf.float32,
                initializer = create_initializer(0.02))
            #shape:[batch,r,r,d]
            v_ = tf.nn.conv1d(input=self.v,filter=W_v)
            """
            s_q, shape is [r,r,d], as the kernal, need to expand to [r,r,d,1]
            conv is [r,r,1]
            """
            conv = tf.nn.conv2d(input=v_,filter=tf.expand_dims(s_q,-1))
            softmax_logits = tf.reshape(conv,shape=[r*r])
            attention_probs = tf.nn.softmax(softmax_logits,aixs=0)
            attention_probs = tf.reshape(attention_probs,shape=[r,r]) #[r,r]
            v_q = tf.nn.avg_pool(
                tf.math.multiply(tf.expand_dims(attention_probs,-1),v_),
                ksize=[1,r,r,1],
                strides=[1,1,1,1])
            return v_q

    def build_LAN(self,config):
        with tf.varibale_scope("LAN"): 
            q_shape = tf.shape(self.q)
            K_shape = tf.shape(self.K)
            dq = q_shape[1]
            dk = K_shape[2]
            d = config["d"]
            W_ql = tf.get_variable(
                name = "W_ql",
                shape = [dq,d],
                dtype = tf.float32,
                initializer = create_initializer(0.02))
            W_kl = tf.get_varialbe(
                name = "W_kl",
                shape = [dk,d],
                dtype = tf.float32,
                initializer = create_initializer(0.02))
            W_l = tf.get_variable(
                name = "W_l",
                shape = [d,d],
                dtype = tf.float32,
                initializer = create_initializer(0.02))
            # attention_key shape:[k,d], k is the number of Key
            attention_key = tf.matmul(self.K, W_kl)
            # attention_scores shape:[1,k]
            attention_scores = tf.matmul(tf.matmul(tf.matmul(self.q,W_ql),W_l), tf.transpose(attention_key))
            attention_probs = tf.nn.softmax(attention_scores)
            # [k,1] multiply [k,d]
            k_q = tf.math.multiply(tf.transpose(attention_probs),attention_key)
            return k_q

    def build_MAT(self,config):
        with tf.variable_scope("MTN"):
            q_shape = tf.shape(self.q)
            dq = q_shape[1]
            d = config["d"]
            W_qm = tf.get_variable(
                name = "W_qm",
                shape = [dq,d],
                dtype = tf.float32,
                initializer = create_initializer(0.02))
            b_qm = tf.get_variable(
                name = "b_qm",
                shape = [1,d],
                dtype = tf.float32,
                initializer = create_initializer(0.02))
            q_m = tf.nn.relu(tf.add(tf.matmul(self.q,W_qm),b_qm))
            W_qm_ = tf.get_variable(
                name = "W_qm_",
                shape = [dq,d],
                dtype = tf.float32,
                initializer = create_initializer(0.02))
            b_qm_ = tf.get_variable(
                name = "b_qm_",
                shape = [1,d],
                dtype = tf.float32,
                initializer = create_initializer(0.02))
            q_ = tf.nn.relu(tf.add(tf.matmul(self.q,W_qm_),b_qm_))

            c_v = get_cosine_distance(q_,self.v_q)
            c_k = get_cosine_distance(q_,self.k_q)
            p_v = tf.nn.softmax(c_v)
            p_k = tf.nn.softmax(c_k)
            x_q = tf.multiply(p_v,self.v_q) + tf.multiply(p_k,self.k_q)
            return q_m,x_q

    def get_loss(self):
        return self.loss
    
    def get_predict(self):
        cosin_dist = get_cosine_distance(self.q_m,self.x_q);
        return cosin_dist

def get_cosine_distance(a_matrix,b_matrix):
    norm1 = tf.sqrt(tf.reduce_sum(tf.square(a_matrix),axis=-1))
    norm2 = tf.sqrt(tf.reduce_sum(tf.square(b_matrix),axis=-1))
    iner_dot = tf.reduce_sum(tf.multiply(a_matrix,b_matrix),axis = -1)
    cosin = tf.divide(iner_dot,(norm1 * norm2))
    return cosin

def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.truncated_normal_initializer(stddev=initializer_range)
                


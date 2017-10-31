import tensorflow as tf
import numpy as np
import random
import math

tf.set_random_seed(123456789)
np.random.seed(123456789)
random.seed(123456789)


def BPR (user_count, item_count, hidden_dim):
    u = tf.placeholder(tf.int32, [None])
    i = tf.placeholder(tf.int32, [None])
    j = tf.placeholder(tf.int32, [None])

    with tf.device("/cpu:0"):
        user_emb_w = tf.get_variable("user_emb_w", [user_count + 1, hidden_dim],
                                     initializer=tf.random_normal_initializer(0, 0.1))
        item_emb_w = tf.get_variable("item_emb_w", [item_count + 1, hidden_dim],
                                     initializer=tf.random_normal_initializer(0, 0.1))
        item_b = tf.get_variable("item_b", [item_count + 1, 1],
                                 initializer=tf.constant_initializer(0.0))
        u_emb = tf.nn.embedding_lookup(user_emb_w, u)
        i_emb = tf.nn.embedding_lookup(item_emb_w, i)
        i_b = tf.nn.embedding_lookup(item_b, i)
        j_emb = tf.nn.embedding_lookup(item_emb_w, j)
        j_b = tf.nn.embedding_lookup(item_b, j)




    rating_mat = tf.matmul(user_emb_w, tf.transpose(item_emb_w))

    # MF predict: u_i > u_j
    x = i_b - j_b + tf.reduce_sum(tf.multiply(u_emb, (i_emb - j_emb)), 1, keep_dims=True)

    # AUC for one user:
    # reasonable iff all (u,i,j) pairs are from the same user
    #
    # average AUC = mean( auc for each user in test set)
    mf_auc = tf.reduce_mean(tf.to_float(x > 0))

    l2_norm = tf.add_n([
        tf.reduce_sum(tf.multiply(u_emb, u_emb)),
        tf.reduce_sum(tf.multiply(i_emb, i_emb)),
        tf.reduce_sum(tf.multiply(j_emb, j_emb))
    ])

    regulation_rate = 0.0001
    bprloss = regulation_rate * l2_norm - tf.reduce_mean(tf.log(tf.sigmoid(x)))

    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(bprloss)

    return u, i, j, mf_auc, bprloss, train_op, rating_mat

# def SVD():


# def SVDpp():



def Autorec (itemCount, h, r): 
    data = tf.placeholder(tf.float32, [None, itemCount])
    mask = tf.placeholder(tf.float32, [None, itemCount])

    scale = math.sqrt(6 / (itemCount + h))

    W1 = tf.Variable(tf.random_uniform([itemCount, h], -scale, scale))
    b1 = tf.Variable(tf.random_uniform([h], -scale, scale))
    mid = tf.nn.sigmoid(tf.matmul(data, W1) + b1)

    W2 = tf.Variable(tf.random_uniform([h, itemCount], -scale, scale))
    b2 = tf.Variable(tf.random_uniform([itemCount], -scale, scale))
    y = tf.matmul(mid, W2) + b2

    regularizer = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(b1) + tf.nn.l2_loss(b2)

    cost = tf.reduce_mean(tf.reduce_sum(tf.square((y - data) * mask), 1, keep_dims=True)
                          + r * regularizer)          
    return data, mask, y, cost



def DualAPR_tanh (itemCount, h, r, ran, alpha, beta):
    data_e = tf.placeholder(tf.float32, [None, itemCount])
    mask_e = tf.placeholder(tf.float32, [None, itemCount])
    data_i = tf.placeholder(tf.float32, [None, itemCount])
    mask_i = tf.placeholder(tf.float32, [None, itemCount])

    #length_negative = tf.placeholder(tf.float32)
    #length_positive = tf.placeholder(tf.float32)
    length_samples = tf.placeholder(tf.float32)
    scale = math.sqrt(6 / (itemCount + h))

    # explicit rating network
    W1_e = tf.Variable(tf.random_uniform([itemCount, h], -scale, scale))
    b1_e = tf.Variable(tf.random_uniform([h], -scale, scale))
    mid_e = tf.nn.sigmoid(tf.matmul(data_e, W1_e) + b1_e)

    W2_e = tf.Variable(tf.random_uniform([h, itemCount], -scale, scale))
    b2_e = tf.Variable(tf.random_uniform([itemCount], -scale, scale))
    y_e = tf.matmul(mid_e, W2_e) + b2_e

    # implicit feedback network
    W1_i = tf.Variable(tf.random_uniform([itemCount, h], -scale, scale))
    b1_i = tf.Variable(tf.random_uniform([h], -scale, scale))
    mid_i = tf.nn.sigmoid(tf.matmul(data_i, W1_i) + b1_i)

    W2_i = tf.Variable(tf.random_uniform([h, itemCount], -scale, scale))
    b2_i = tf.Variable(tf.random_uniform([itemCount], -scale, scale))
    y_i = tf.matmul(mid_i, W2_i) + b2_i
    
    # tanh, (e^(20(x-0.5))-1) / (e^(20(x-0.5))+1)
    approx_unintMask = tf.tanh( alpha*(y_i - beta) )
    
    AE_loss_i = tf.reduce_mean(tf.reduce_sum(tf.square((y_i - data_i) * mask_i), 1, keep_dims=True))
    AE_loss_e = tf.reduce_mean(tf.reduce_sum(tf.square((y_e - data_e) * mask_e), 1, keep_dims=True))
    RANK_loss = -(1/length_samples) * tf.reduce_mean(tf.reduce_sum(y_e*approx_unintMask, 1, keep_dims=True))
    regularizer = tf.reduce_mean(tf.nn.l2_loss(W1_e) + tf.nn.l2_loss(W2_e) + tf.nn.l2_loss(b1_e) + tf.nn.l2_loss(b2_e)+tf.nn.l2_loss(W1_i) + tf.nn.l2_loss(W2_i) + tf.nn.l2_loss(b1_i) + tf.nn.l2_loss(b2_i))

    cost = tf.reduce_mean(AE_loss_i + AE_loss_e + ran*RANK_loss + r*regularizer)

    return data_e, mask_e, data_i, mask_i, length_samples, y_e, cost    

def APR (itemCount, h, r, ran): 
    data = tf.placeholder(tf.float32, [None, itemCount])
    mask = tf.placeholder(tf.float32, [None, itemCount])
  
    negativeMask = tf.placeholder(tf.float32, [None, itemCount])
    positiveMask = tf.placeholder(tf.float32, [None, itemCount])
    
    length_negative = tf.placeholder(tf.float32)
    length_positive = tf.placeholder(tf.float32)
    
    scale = math.sqrt(6 / (itemCount + h))

    W1 = tf.Variable(tf.random_uniform([itemCount, h], -scale, scale))
    b1 = tf.Variable(tf.random_uniform([h], -scale, scale))
    mid = tf.nn.sigmoid(tf.matmul(data, W1) + b1)

    W2 = tf.Variable(tf.random_uniform([h, itemCount], -scale, scale))
    b2 = tf.Variable(tf.random_uniform([itemCount], -scale, scale))
    
    y = tf.matmul(mid, W2) + b2

    AE_loss = tf.reduce_mean(tf.reduce_sum(tf.square((y - data) * mask), 1, keep_dims=True))
    RANK_loss = (1/length_positive)*tf.reduce_mean(tf.reduce_sum(y*negativeMask, 1, keep_dims=True)) - (1/length_negative)*tf.reduce_mean(tf.reduce_sum(y*positiveMask, 1, keep_dims=True))
    regularizer = tf.reduce_mean(tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(b1) + tf.nn.l2_loss(b2))
    
    cost = tf.reduce_mean(AE_loss + ran*RANK_loss + r*regularizer)

    return data, mask, positiveMask, negativeMask, length_positive, length_negative, y, cost

    
def DualAPR_sigmoid (itemCount, h, r, ran, alpha, beta): 
    data_e = tf.placeholder(tf.float32, [None, itemCount])
    mask_e = tf.placeholder(tf.float32, [None, itemCount])
    data_i = tf.placeholder(tf.float32, [None, itemCount])
    mask_i = tf.placeholder(tf.float32, [None, itemCount])
    
    positiveMask = tf.placeholder(tf.float32, [None, itemCount])
    
    #length_negative = tf.placeholder(tf.float32)
    #length_positive = tf.placeholder(tf.float32)
    length_samples = tf.placeholder(tf.float32)
    scale = math.sqrt(6 / (itemCount + h))

    # explicit rating network
    W1_e = tf.Variable(tf.random_uniform([itemCount, h], -scale, scale))
    b1_e = tf.Variable(tf.random_uniform([h], -scale, scale))
    mid_e = tf.nn.sigmoid(tf.matmul(data_e, W1_e) + b1_e)

    W2_e = tf.Variable(tf.random_uniform([h, itemCount], -scale, scale))
    b2_e = tf.Variable(tf.random_uniform([itemCount], -scale, scale))
    y_e = tf.matmul(mid_e, W2_e) + b2_e

    # implicit feedback network
    W1_i = tf.Variable(tf.random_uniform([itemCount, h], -scale, scale))
    b1_i = tf.Variable(tf.random_uniform([h], -scale, scale))
    mid_i = tf.nn.sigmoid(tf.matmul(data_i, W1_i) + b1_i)

    W2_i = tf.Variable(tf.random_uniform([h, itemCount], -scale, scale))
    b2_i = tf.Variable(tf.random_uniform([itemCount], -scale, scale))
    y_i = tf.matmul(mid_i, W2_i) + b2_i
    
    # hard sigmoid, 1-(1/(1+1/e^(20(x-0.7))))
    approx_unintMask = 1 - tf.sigmoid( alpha*(y_i - beta) )
    AE_loss_i = tf.reduce_mean(tf.reduce_sum(tf.square((y_i - data_i) * mask_i), 1, keep_dims=True))
    AE_loss_e = tf.reduce_mean(tf.reduce_sum(tf.square((y_e - data_e) * mask_e), 1, keep_dims=True))

    RANK_loss = (-1/length_samples)*(tf.reduce_mean(tf.reduce_sum(y_e*positiveMask, 1, keep_dims=True)) - tf.reduce_mean(tf.reduce_sum(y_e*approx_unintMask, 1, keep_dims=True)))
    
    regularizer = tf.reduce_mean(tf.nn.l2_loss(W1_e) + tf.nn.l2_loss(W2_e) + tf.nn.l2_loss(b1_e) + tf.nn.l2_loss(b2_e)+tf.nn.l2_loss(W1_i) + tf.nn.l2_loss(W2_i) + tf.nn.l2_loss(b1_i) + tf.nn.l2_loss(b2_i))
    
    cost = tf.reduce_mean(AE_loss_i + AE_loss_e + ran*RANK_loss + r*regularizer)

    return data_e, mask_e, data_i, mask_i, positiveMask, length_samples, y_e, cost    

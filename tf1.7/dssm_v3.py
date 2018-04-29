"""
train pack size and test pack size are total rows amount devide by 1024, our batch to back propagation is 1024
"""

import pickle
import random
import time, scipy
import scipy.sparse
import sys, os
import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('datadir', '.', 'Data Directory')
flags.DEFINE_string('outputdir', '.', 'Model Directory')
flags.DEFINE_string('logdir', '/tmp/dssm-400-120-relu', 'Summaries directory')
flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 900000, 'Number of steps to run trainer.')
flags.DEFINE_integer('epoch_steps', 18000, "Number of steps in one epoch.")
flags.DEFINE_bool('gpu', 1, "Enable GPU or not")

start = time.time()

doc_train_data = None
query_train_data = None

# load test data for now
query_test_data = scipy.sparse.load_npz(os.path.join(FLAGS.datadir,'query.test.npz')).tocsr()
doc_test_data = scipy.sparse.load_npz(os.path.join(FLAGS.datadir,'doc.test.npz')).tocsr()
print("query test rows:%s, doc test rows:%s"%(query_test_data.shape[0], doc_test_data.shape[0]))
doc_train_data = scipy.sparse.load_npz(os.path.join(FLAGS.datadir,'doc.train.npz')).tocsr()
query_train_data = scipy.sparse.load_npz(os.path.join(FLAGS.datadir,'query.train.npz')).tocsr()
print("query train rows:%s, doc train rows:%s"%(query_train_data.shape[0], doc_train_data.shape[0]))
train_pack_size = int(query_train_data.shape[0] / 1024)
test_pack_size = int(query_test_data.shape[0] / 1024)
print("test pack size:%s, train pack size:%s"%(test_pack_size, train_pack_size))

if train_pack_size == 0 or test_pack_size == 0:
    print("either train data rows or test data rows are less than 1024")
    sys.exit(-1)

end = time.time()
print("Loading data from HDD to memory: %.2fs" % (end - start))

TRIGRAM_D = 49284

NEG = 50
BS = 1024

L1_N = 400
L2_N = 120

query_in_shape = np.array([BS, TRIGRAM_D], np.int64)
doc_in_shape = np.array([BS, TRIGRAM_D], np.int64)


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.summary.scalar('sttdev/' + name, stddev)
        tf.summary.scalar('max/' + name, tf.reduce_max(var))
        tf.summary.scalar('min/' + name, tf.reduce_min(var))
        tf.summary.histogram(name, var)


with tf.name_scope('input'):
    # Shape [BS, TRIGRAM_D].
    query_batch = tf.sparse_placeholder(tf.float32, name='QueryBatch')
    # Shape [BS, TRIGRAM_D]
    doc_batch = tf.sparse_placeholder(tf.float32, name='DocBatch')

with tf.name_scope('L1'):
    l1_par_range = np.sqrt(6.0 / (TRIGRAM_D + L1_N))
    weight1 = tf.Variable(tf.random_uniform([TRIGRAM_D, L1_N], -l1_par_range, l1_par_range))
    bias1 = tf.Variable(tf.random_uniform([L1_N], -l1_par_range, l1_par_range))
    variable_summaries(weight1, 'L1_weights')
    variable_summaries(bias1, 'L1_biases')

    # query_l1 = tf.matmul(tf.to_float(query_batch),weight1)+bias1
    query_l1 = tf.sparse_tensor_dense_matmul(query_batch, weight1) + bias1
    # doc_l1 = tf.matmul(tf.to_float(doc_batch),weight1)+bias1
    doc_l1 = tf.sparse_tensor_dense_matmul(doc_batch, weight1) + bias1

    query_l1_out = tf.nn.relu(query_l1)
    doc_l1_out = tf.nn.relu(doc_l1)

with tf.name_scope('L2'):
    l2_par_range = np.sqrt(6.0 / (L1_N + L2_N))

    weight2 = tf.Variable(tf.random_uniform([L1_N, L2_N], -l2_par_range, l2_par_range))
    bias2 = tf.Variable(tf.random_uniform([L2_N], -l2_par_range, l2_par_range))
    variable_summaries(weight2, 'L2_weights')
    variable_summaries(bias2, 'L2_biases')

    query_l2 = tf.matmul(query_l1_out, weight2) + bias2
    doc_l2 = tf.matmul(doc_l1_out, weight2) + bias2
    query_y = tf.nn.relu(query_l2)
    doc_y = tf.nn.relu(doc_l2)

with tf.name_scope('FD_rotate'):
    # Rotate FD+ to produce 50 FD-
    temp = tf.tile(doc_y, [1, 1])

    for i in range(NEG):
        rand = int((random.random() + i) * BS / NEG)
        doc_y = tf.concat([doc_y,
                           tf.slice(temp, [rand, 0], [BS - rand, -1]),
                           tf.slice(temp, [0, 0], [rand, -1])], 0)

with tf.name_scope('Cosine_Similarity'):
    # Cosine similarity
    query_norm = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(query_y), 1, True)), [NEG + 1, 1])
    doc_norm = tf.sqrt(tf.reduce_sum(tf.square(doc_y), 1, True))

    prod = tf.reduce_sum(tf.multiply(tf.tile(query_y, [NEG + 1, 1]), doc_y), 1, True)
    norm_prod = tf.multiply(query_norm, doc_norm)

    cos_sim_raw = tf.truediv(prod, norm_prod)
    cos_sim = tf.transpose(tf.reshape(tf.transpose(cos_sim_raw), [NEG + 1, BS])) * 20

with tf.name_scope('Loss'):
    # Train Loss
    # prob BS * 51 matrix
    # prob = tf.nn.softmax((cos_sim))
    # y BS * 51 matrix too
    label_value = np.array([[1] + [0] * NEG] * BS)
    loss = tf.losses.softmax_cross_entropy(onehot_labels = label_value, logits = cos_sim + np.finfo(np.float32).eps)
    tf.summary.scalar('loss', loss)

with tf.name_scope('Evaluate'):
    prob = tf.nn.softmax(cos_sim)
    hitprob = tf.slice(prob, [0, 0], [-1, 1], 'hitprob')

with tf.name_scope('Training'):
    # Optimizer
    train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss)

# with tf.name_scope('Accuracy'):
#     correct_prediction = tf.equal(tf.argmax(prob, 1), 0)
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()

with tf.name_scope('Test'):
    average_loss = tf.placeholder(tf.float32)
    loss_summary = tf.summary.scalar('average_loss', average_loss)


def pull_batch(query_data, doc_data, batch_idx):
    # start = time.time()
    query_in = query_data[batch_idx * BS:(batch_idx + 1) * BS, :]
    doc_in = doc_data[batch_idx * BS:(batch_idx + 1) * BS, :]
    
    if batch_idx == 0:
      print(query_in.getrow(53))
    query_in = query_in.tocoo()
    doc_in = doc_in.tocoo()
    
    

    query_in = tf.SparseTensorValue(
        np.transpose([np.array(query_in.row, dtype=np.int64), np.array(query_in.col, dtype=np.int64)]),
        np.array(query_in.data, dtype=np.float),
        np.array(query_in.shape, dtype=np.int64))
    doc_in = tf.SparseTensorValue(
        np.transpose([np.array(doc_in.row, dtype=np.int64), np.array(doc_in.col, dtype=np.int64)]),
        np.array(doc_in.data, dtype=np.float),
        np.array(doc_in.shape, dtype=np.int64))
    print("pulled_query_in.shape:%s"%(query_in.dense_shape))
    print("pulled_doc_in.shape:%s"%(doc_in.dense_shape))

    # end = time.time()
    # print("Pull_batch time: %f" % (end - start))

    return query_in, doc_in


def feed_dict(Train, batch_idx):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if Train:
        query_in, doc_in = pull_batch(query_train_data, doc_train_data, batch_idx)
    else:
        query_in, doc_in = pull_batch(query_test_data, doc_test_data, batch_idx)
    return {query_batch: query_in, doc_batch: doc_in}


config = tf.ConfigProto()  # log_device_placement=True)
config.gpu_options.allow_growth = True
#if not FLAGS.gpu:
#config = tf.ConfigProto(device_count= {'GPU' : 0})
saver = tf.train.Saver()

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(os.path.join(FLAGS.logdir, 'train'), sess.graph)
    test_writer = tf.summary.FileWriter(os.path.join(FLAGS.logdir, 'test'), sess.graph)

    # Actual execution
    start = time.time()

    for step in range(FLAGS.max_steps):
        batch_idx = step % FLAGS.epoch_steps

        if batch_idx % (train_pack_size / 64) == 0:
            progress = 100.0 * batch_idx / FLAGS.epoch_steps
            print("\r%.2f%% Epoch" % progress)
            

        sess.run(train_step, feed_dict=feed_dict(True, batch_idx % train_pack_size))

        if batch_idx == FLAGS.epoch_steps - 1:
            end = time.time()
            epoch_loss = 0
            for i in range(train_pack_size):
                loss_v = sess.run(loss, feed_dict=feed_dict(True, i))
                epoch_loss += loss_v

            epoch_loss /= train_pack_size
            train_loss = sess.run(loss_summary, feed_dict={average_loss: epoch_loss})
            train_writer.add_summary(train_loss, step + 1)

            print ("\nEpoch #%-5d | Train Loss: %-4.3f | PureTrainTime: %-3.3fs" %(step / FLAGS.epoch_steps, epoch_loss, end - start))
            epoch_loss = 0
            for i in range(test_pack_size):
                loss_v = sess.run(loss, feed_dict=feed_dict(False, i))
                epoch_loss += loss_v

            epoch_loss /= test_pack_size

            test_loss = sess.run(loss_summary, feed_dict={average_loss: epoch_loss})
            test_writer.add_summary(test_loss, step + 1)

            start = time.time()
            print ("Epoch #%-5d | Test  Loss: %-4.3f | Calc_LossTime: %-3.3fs" %
                   (step / FLAGS.epoch_steps, epoch_loss, start - end))
    saver.save(sess, os.path.join(FLAGS.outputdir, 'my-model'))

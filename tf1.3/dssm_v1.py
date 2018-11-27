import random
import time,os
import scipy
import sys
import numpy as np
import tensorflow as tf
import argparse
import os
from scipy.sparse import coo_matrix

parser = argparse.ArgumentParser(description='DSSM on tensorflow')
parser.add_argument('--querytest', type=str, default= 'query.test.npz',  help='Name of the query test data')
parser.add_argument('--doctest', type=str, default= 'doc.test.npz',  help='Name of the doc test data')
parser.add_argument('--querytrain', type=str, default= 'query.train.npz',  help='Name of the query train data')
parser.add_argument('--doctrain', type=str, default= 'doc.train.npz',  help='Name of the doc train data')

parser.add_argument('--datadir', '--input-training-data-path', type=str, default= './',  help='Path of the train/test data')
parser.add_argument('--modeldir', '--output-model-path', type=str, default= './',  help='Output path of the doc train data')
parser.add_argument('--summariesdir', type=str, default= './',  help='Path of the doc train data')
parser.add_argument('--logdir', type=str, default= './',  help='Path of the doc train data')

parser.add_argument('--learningrate', type=float, default=0.001,  help='Initial learning rate.')

parser.add_argument('--wholeitr', type=int, default=50,  help='Number of steps to run trainer.')
parser.add_argument('--epochitr', type=int, default=10,  help='Number of steps in one epoch when traning')

parser.add_argument('--trainpacksize', type=int, default=1300,  help='Number of train batches in one pickle pack')
parser.add_argument('--testpacksize', type=int, default=1300,  help='Number of test batches in one pickle pack')

parser.add_argument('--negativesize', type=int, default=50,  help='Number of negativesize')

parser.add_argument('--optimizer', type=str, default='tf.train.GradientDescentOptimizer',  help='optimizer type')
parser.add_argument('--lossfunction', type=str, default='tf.log',  help='lossfunction type')

parser.add_argument('--saveperepoch', type=str, default='1',  help='wheather save model every epochitr')
parser.add_argument('--enablesummary', type=str, default='1',  help='wheather enable summary')
parser.add_argument('--enabledebug', type=str, default='0',  help='wheather enable debug')

parser.add_argument('--isphilly', type=str, default='0',  help='environment')

args, unknown = parser.parse_known_args()
args.logdir = args.modeldir
args.summariesdir = args.modeldir

if args.isphilly == '1':
    SUMMARIESDIR = os.path.join(args.summariesdir, "\\..\\..\\")
else:
    SUMMARIESDIR = os.path.join(args.summariesdir, "tensorboard")
MODELPATH = os.path.join(args.modeldir, "model")

if not os.path.exists(SUMMARIESDIR):
    os.makedirs(SUMMARIESDIR)

if not os.path.exists(MODELPATH):
    os.makedirs(MODELPATH)

start = time.time()

TRIGRAM_D = 49284
BS = 1024

L1_N = 300
L2_N = 300
L3_N = 180

NEG = max(50, args.negativesize)

query_in_shape = shape=[None,TRIGRAM_D]
doc_in_shape = shape=[None,TRIGRAM_D]

ACTIVATION = tf.nn.relu

LOSSFUNCTION = args.lossfunction

SHARENETWORK = True

if args.optimizer == 'tf.train.AdamOptimizer':
    OPTIMIZER = tf.train.AdamOptimizer
else:
    OPTIMIZER = tf.train.GradientDescentOptimizer

TRAIN_SUMMARY_KEY = 'train'
LOSS_SUMMARY_KEY = 'loss'
EVA_SUMMARY_KEY = 'evaluate'

RIGHTLABELS = np.array([0]*BS)

doc_train_data = None
query_train_data = None

# load test data for now
query_test_data = scipy.sparse.load_npz(os.path.join(args.datadir, args.querytest)).tocsr()
doc_test_data = scipy.sparse.load_npz(os.path.join(args.datadir, args.doctest)).tocsr()

doc_train_data = scipy.sparse.load_npz(os.path.join(args.datadir, args.querytrain)).tocsr()
query_train_data = scipy.sparse.load_npz(os.path.join(args.datadir, args.doctest)).tocsr()

train_pack_size = int(query_train_data.shape[0] / BS)
test_pack_size = int(query_test_data.shape[0] / BS)

if train_pack_size == 0 or test_pack_size == 0:
    print("either train data rows or test data rows are less than 1024")
    sys.exit(-1)

end = time.time()
print("Loading data from HDD to memory: %.2fs\n" % (end - start))

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('learning_rate', args.learningrate, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 0, 'Number of steps to run trainer.')
flags.DEFINE_integer('epoch_steps', 0, "Number of steps in one epoch.")
flags.DEFINE_integer('train_pack_size', 0, "Number of batches in one pack.")
flags.DEFINE_integer('test_pack_size', 0, "Number of batches in one pack.")
flags.DEFINE_bool('gpu', 1, "Enable GPU or not")

FLAGS.train_pack_size = min(args.trainpacksize, train_pack_size)
FLAGS.test_pack_size = min(args.testpacksize, test_pack_size)
FLAGS.epoch_steps = FLAGS.train_pack_size * args.epochitr
FLAGS.max_steps = FLAGS.epoch_steps * args.wholeitr

def network_summary():
    print("ACTIVATION: {0}\tOPTIMIZER: {1}\tLOSSFUNCTION: {2}\tSHARENETWORK: {3}\n".format(ACTIVATION.__name__,OPTIMIZER.__name__, LOSSFUNCTION, str(SHARENETWORK)))
    print("TRIGRAM_D: {0}\tBS: {1}\tL1_N/L2_N/L3_N: {2}/{3}/{4}\n".format(TRIGRAM_D, BS, L1_N, L2_N, L3_N))
    print("in_train_pack_size: {0}\targ_train_pack_size: {1}\treal_train_pack_size: {2}\n".format(train_pack_size, args.trainpacksize, FLAGS.train_pack_size))
    print("in_test_pack_size: {0}\targ_test_pack_size: {1}\treal_test_pack_size: {2}\n".format(test_pack_size, args.testpacksize, FLAGS.test_pack_size))
    print("epochitr: {0}\twholeitr: {1}\n".format(args.epochitr, args.wholeitr))
    print("epoch_steps: {0}\tmax_steps: {1}\n".format(FLAGS.epoch_steps, FLAGS.max_steps))
    print("learning_rate: {0}\n".format(FLAGS.learning_rate))
    print("saveperepoch: {0}\tenablesummary: {1}".format(args.saveperepoch, args.enablesummary))

def variable_summaries(var, name, key=[TRAIN_SUMMARY_KEY]):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean, collections = key)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.summary.scalar('sttdev/' + name, stddev, collections = key)
        tf.summary.scalar('max/' + name, tf.reduce_max(var), collections = key)
        tf.summary.scalar('min/' + name, tf.reduce_min(var), collections = key)
        tf.summary.histogram(name, var, collections = key)

def batch_normalization(x, phase_train, out_size):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        out_size:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[out_size]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[out_size]),
                            name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5, name='moving')

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def add_layer_x(inputs, in_size, out_size, layer_name, phase_train,  sparse_matmul=False, norm=False, activation_function=None):
    wlimit = np.sqrt(6.0 / (in_size + out_size))
    Weights = tf.Variable(tf.random_uniform([in_size, out_size], -wlimit, wlimit), name='weights')
    biases = tf.Variable(tf.random_uniform([out_size], -wlimit, wlimit), name='biases')

    variable_summaries(Weights, layer_name +'_weights')
    variable_summaries(biases, layer_name + '_biases')

    if sparse_matmul:
        wx_plus_b = tf.sparse_tensor_dense_matmul(inputs, Weights) + biases
    else:
        wx_plus_b = tf.matmul(inputs, Weights) + biases

    wx_plus_b = batch_normalization(wx_plus_b, phase_train, out_size)

    if activation_function is None:
        outputs = wx_plus_b
    else:
        outputs = activation_function(wx_plus_b)
    return outputs

# share weight/bias on x/y
def add_layer_x_y(inputs_x, inputs_y, in_size, out_size, layer_name, phase_train,  sparse_matmul=False, norm=False, activation_function=None):
    wlimit = np.sqrt(6.0 / (in_size + out_size))
    Weights = tf.Variable(tf.random_uniform([in_size, out_size], -wlimit, wlimit), name='weights')
    biases = tf.Variable(tf.random_uniform([out_size], -wlimit, wlimit), name='biases')

    variable_summaries(Weights, layer_name + '_weights')
    variable_summaries(biases, layer_name + '_biases')

    if sparse_matmul:
        wx_plus_b = tf.sparse_tensor_dense_matmul(inputs_x, Weights) + biases
        wy_plus_b = tf.sparse_tensor_dense_matmul(inputs_y, Weights) + biases
    else:
        wx_plus_b = tf.matmul(inputs_x, Weights) + biases
        wy_plus_b = tf.matmul(inputs_y, Weights) + biases

    variable_summaries(wx_plus_b, layer_name + '_wx_plus_b')
    variable_summaries(wy_plus_b, layer_name + '_wy_plus_b')

    wx_plus_b_bn = batch_normalization(wx_plus_b, phase_train, out_size)
    wy_plus_b_bn = batch_normalization(wy_plus_b, phase_train, out_size)

    variable_summaries(wx_plus_b_bn, layer_name + '_wx_plus_b_bn')
    variable_summaries(wy_plus_b_bn, layer_name + '_wy_plus_b_bn')

    if activation_function is None:
        outputs_x = wx_plus_b_bn
        outputs_y = wy_plus_b_bn
    else:
        outputs_x = activation_function(wx_plus_b_bn)
        outputs_y = activation_function(wy_plus_b_bn)

    variable_summaries(outputs_x, layer_name + '_outputs_x')
    variable_summaries(outputs_y, layer_name + '_outputs_y')

    return outputs_x, outputs_y

with tf.name_scope('input'):
    # Shape [BS, TRIGRAM_D].
    query_batch = tf.sparse_placeholder(tf.float32, shape=query_in_shape, name='QueryBatch')
    # Shape [BS, TRIGRAM_D]
    doc_batch = tf.sparse_placeholder(tf.float32, shape=doc_in_shape, name='DocBatch')

    on_train = tf.placeholder(tf.bool, shape=[], name='OnTrain')

with tf.name_scope('L1'):
    # share the common weight/bias for doc/query?

    if SHARENETWORK:
        query_l1_out, doc_l1_out = add_layer_x_y(
            query_batch,
            doc_batch,
            TRIGRAM_D,
            L1_N,
            'L1',
            phase_train = on_train,
            sparse_matmul=True,
            norm=True,
            activation_function=ACTIVATION)
    else:
        query_l1_out = add_layer_x(
            query_batch,
            TRIGRAM_D,
            L1_N,
            'L1',
            phase_train = on_train,
            sparse_matmul=True,
            norm=True,
            activation_function=ACTIVATION)
        doc_l1_out = add_layer_x(
            doc_batch,
            TRIGRAM_D,
            L1_N,
            'L1',
            phase_train = on_train,
            sparse_matmul=True,
            norm=True,
            activation_function=ACTIVATION)


with tf.name_scope('L2'):
    if SHARENETWORK:
            query_l2_out, doc_l2_out = add_layer_x_y(
            query_l1_out,
            doc_l1_out,
            L1_N,
            L2_N,
            'L2',
            phase_train = on_train,
            sparse_matmul=False,
            norm=True,
            activation_function=ACTIVATION)
    else:
        query_l2_out = add_layer_x(
            query_l1_out,
            L1_N,
            L2_N,
            'L2',
            phase_train = on_train,
            sparse_matmul=False,
            norm=True,
            activation_function=ACTIVATION)
        doc_l2_out = add_layer_x(
            doc_l1_out,
            L1_N,
            L2_N,
            'L2',
            phase_train = on_train,
            sparse_matmul=False,
            norm=True,
            activation_function=ACTIVATION)

with tf.name_scope('L3'):
    if SHARENETWORK:
            query_y, doc_y = add_layer_x_y(
            query_l2_out,
            doc_l2_out,
            L2_N,
            L3_N,
            'L3',
            phase_train = on_train,
            sparse_matmul=False,
            norm=True,
            activation_function=ACTIVATION)
    else:
        query_y = add_layer_x(
            query_l2_out,
            L2_N,
            L3_N,
            'L3',
            phase_train = on_train,
            sparse_matmul=False,
            norm=True,
            activation_function=ACTIVATION)
        doc_y = add_layer_x(
            doc_l2_out,
            L2_N,
            L3_N,
            'L3',
            phase_train = on_train,
            sparse_matmul=False,
            norm=True,
            activation_function=ACTIVATION)

with tf.name_scope('FD_rotate'):
    # Rotate FD+ to produce 50 FD-
    temp = tf.tile(doc_y, [1, 1])

    for i in range(NEG):
        rand = int((random.random() + i) * BS / NEG)
        doc_y = tf.concat([doc_y, tf.slice(temp, [rand, 0], [BS - rand, -1]), tf.slice(temp, [0, 0], [rand, -1])], 0)

with tf.name_scope('Cosine_Similarity'):
    # Cosine similarity
    query_norm = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(query_y, name='debug_1'), 1, True, name='debug_1_2'), name='debug_1_3'), [NEG + 1, 1], name='debug_1_4')
    doc_norm = tf.sqrt(tf.reduce_sum(tf.square(doc_y, name='debug_2_1'), 1, True, name='debug_2_2'), name='debug_2_3')
    prod = tf.reduce_sum(tf.multiply (tf.tile(query_y, [NEG + 1, 1], name='debug_3_1'), doc_y, name='debug_3_2'), 1, True, name='debug_3_3')
    norm_prod = tf.multiply (query_norm, doc_norm, name='debug_4_1')

    cos_sim_raw = tf.truediv(prod, norm_prod, name='debug_5_1')
    cos_sim = tf.transpose(tf.reshape(tf.transpose(cos_sim_raw, name='debug_6_1'), [NEG + 1, BS], name='debug_6_2'), name='debug_6_3') * 20
    #print(query_y.shape, doc_y.shape)

with tf.name_scope('Loss'):
    # Train Loss
    if LOSSFUNCTION == 'tf.log':
        prob = tf.nn.softmax(cos_sim)
        hit_prob = tf.clip_by_value(tf.slice(prob, [0, 0], [-1, 1]), 1e-10, 1.0)
        loss = -tf.reduce_sum(tf.log(hit_prob)) / BS
        variable_summaries(hit_prob, 'hit_prob')
    else:
        label_value = np.array([[1] + [0] * NEG] * BS)
        loss = tf.losses.softmax_cross_entropy(onehot_labels = label_value, logits = cos_sim)

        #debug
        prob = tf.nn.softmax(cos_sim)
        hit_prob = tf.clip_by_value(tf.slice(prob, [0, 0], [-1, 1]), 1e-10, 1.0)
        variable_summaries(yesvalue, 'hit_prob')

with tf.name_scope('Training'):
    # Optimizer
    train_step = OPTIMIZER(FLAGS.learning_rate).minimize(loss, name='train_step')

with tf.name_scope('Evaluate'):
    prob = tf.nn.softmax(cos_sim)
    hit_prob = tf.slice(prob, [0, 0], [-1, 1], 'hitprob')
    variable_summaries(hit_prob, 'eva_hit_prob', [EVA_SUMMARY_KEY])

with tf.name_scope('AverageLossSummary'):
    average_loss = tf.placeholder(tf.float32)
    loss_summary = tf.summary.scalar('average_loss', average_loss, collections=[LOSS_SUMMARY_KEY])

def pull_batch(query_data, doc_data, batch_idx):
    query_in = query_data[batch_idx * BS:(batch_idx + 1) * BS, :]
    doc_in = doc_data[batch_idx * BS:(batch_idx + 1) * BS, :]
    
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

    return query_in, doc_in


def feed_dict(on_training, Train, batch_idx):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if Train:
        query_in, doc_in = pull_batch(query_train_data, doc_train_data, batch_idx)
    else:
        query_in, doc_in = pull_batch(query_test_data, doc_test_data, batch_idx)
    return {query_batch: query_in, doc_batch: doc_in, on_train: on_training}

config = tf.ConfigProto()  # log_device_placement=True)
config.gpu_options.allow_growth = True
#if not FLAGS.gpu:
#config = tf.ConfigProto(device_count= {'GPU' : 0})
network_summary()

merged_train = tf.summary.merge_all(key=TRAIN_SUMMARY_KEY)
merged_loss = tf.summary.merge_all(key=LOSS_SUMMARY_KEY)

with tf.Session(config=config) as sess:
    saver = tf.train.Saver(max_to_keep=0)
    sess.run(tf.global_variables_initializer())
    #print("global_variables:\n")
    #print([(str(i.name), i.initial_value) for i in tf.global_variables()])
    #print("local_variables:\n")
    #print([(str(i.name), i.initial_value) for i in tf.local_variables()])
    
    train_writer = tf.summary.FileWriter(os.path.join(SUMMARIESDIR , 'train'), sess.graph)
    test_writer = tf.summary.FileWriter(os.path.join(SUMMARIESDIR , 'test'), sess.graph)

    # Actual execution
    start = time.time()

    cur_epoch_itr = 0
    
    try:
        for step in range(FLAGS.max_steps):
            batch_idx = step % FLAGS.epoch_steps

            if batch_idx % (FLAGS.epoch_steps / 100) == 0:
                progress = 100.0 * batch_idx / FLAGS.epoch_steps
                sys.stdout.write("\r%.2f%% Epoch | " % progress)
                sys.stdout.write("%d Epoch Step" % batch_idx)
                sys.stdout.flush()

                if args.enablesummary == '1':
                    result = sess.run(merged_train, feed_dict=feed_dict(False, True, batch_idx % FLAGS.train_pack_size))
                    train_writer.add_summary(result, step+1)

                if args.enabledebug == '1':
                    saver.save(sess, os.path.join(MODELPATH, 'model_debug'), global_step=step)

            sess.run(train_step, feed_dict=feed_dict(True, True, batch_idx % FLAGS.train_pack_size))

            if batch_idx == FLAGS.epoch_steps - 1:
                end = time.time()
                epoch_loss = 0
                for i in range(FLAGS.train_pack_size):
                    #loss_v, pro_eval = sess.run([loss, binarylogit], feed_dict=feed_dict(False, True, i))
                    #print(tf.shape(loss), loss_v, pro_eval)
                    loss_v = sess.run(loss, feed_dict=feed_dict(False, True, i))
                    epoch_loss += loss_v

                epoch_loss /= FLAGS.train_pack_size
                sys.stdout.write("\repoch_loss : %.2f" % epoch_loss)
                train_loss = sess.run(merged_loss, feed_dict={average_loss: epoch_loss})

                train_writer.add_summary(train_loss, step + 1)

                print ("\nEpoch #%-5d | Train Loss: %-4.3f | PureTrainTime: %-3.3fs" %
                        (step / FLAGS.epoch_steps, epoch_loss, end - start))

                if args.enabledebug != '1':
                    epoch_loss = 0
                    for i in range(FLAGS.test_pack_size):
                        loss_v = sess.run(loss, feed_dict=feed_dict(False, False, i))
                        epoch_loss += loss_v

                    epoch_loss /= FLAGS.test_pack_size
                    test_loss = sess.run(merged_loss, feed_dict={average_loss: epoch_loss})

                    test_writer.add_summary(test_loss, step + 1)

                    start = time.time()
                    print ("Epoch #%-5d | Test  Loss: %-4.3f | Calc_LossTime: %-3.3fs" %
                           (step / FLAGS.epoch_steps, epoch_loss, start - end))


                # save tmp model
                cur_epoch_itr += 1
                if  args.saveperepoch == '1' or cur_epoch_itr % (args.wholeitr/5) == 0:
                    saver.save(sess, os.path.join(MODELPATH, 'model'), global_step=step+1)
                    print("Save tmp model at step: {0}, epoch_itr: {1}".format(step, cur_epoch_itr))

        # save model
        save_path = saver.save(sess, os.path.join(MODELPATH, 'model_final'))

        print("Model saved in path: %s" % save_path)
        print("Model restored.")
    except Exception as e:
        print(str(e))
        print("ERROR !!!!!!!!!!!!!!!!!!!!!!!!!!!\n")

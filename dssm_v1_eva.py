import random
import time
import sys
import numpy as np
import tensorflow as tf
import argparse
import os
import scipy
from scipy.sparse import coo_matrix

parser = argparse.ArgumentParser(description='DSSM on tensorflow')
parser.add_argument('--datadir', type=str, default= './',  help='Path of the doc train data')
parser.add_argument('--modeldir', type=str, default= './model',  help='Path of the model')
parser.add_argument('--modelname', type=str, default= 'model_final',  help='model name')
parser.add_argument('--summariesdir', '--log-dir', type=str, default= './',  help='Path of the doc train data')
parser.add_argument('--queryeva', type=str, default= 'query.eva.npz',  help='Path of the query eva data')
parser.add_argument('--doceva', type=str, default= 'doc.eva.npz',  help='Path of the doc eva data')
parser.add_argument('--scorelist', type=str, default= 'score_list',  help='score list')

parser.add_argument('--evatext', type=str, help='Path of the eva text data')
parser.add_argument('--evaappendtext', type=str, help='Path of the eva text data after append score')
parser.add_argument('--querypos', type=str,  help='Path of the query positive data')
parser.add_argument('--docpos', type=str,  help='Path of the doc positive data')

args, unknown = parser.parse_known_args()

COSINE_EVA_SUMMARY_KEY = 'cosine_evaluate'

SUMMARIESDIR = os.path.join(args.summariesdir, "tensorboard")

if not os.path.exists(SUMMARIESDIR):
    os.makedirs(SUMMARIESDIR)

doc_train_data = None
query_train_data = None

BS = 1024

start = time.time()
# load eva data for now
#tocsr: Return a copy of this matrix in Compressed Sparse Row format. Duplicate entries will be summed together.
# can also visit by a[x,y]
query_eva_data = scipy.sparse.load_npz(os.path.join(args.datadir, args.queryeva)).tocsr()
doc_eva_data = scipy.sparse.load_npz(os.path.join(args.datadir, args.doceva)).tocsr()

if args.querypos and args.docpos and os.path.exit(args.querypos) and osp.path.exit(docpos):
    query_pos_data = scipy.sparse.load_npz(os.path.join(args.datadir, args.querypos)).tocsr()
    doc_pos_data = scipy.sparse.load_npz(os.path.join(args.datadir, args.docpos)).tocsr()
else:
    query_pos_data = None
    doc_pos_data = None

end = time.time()

print("Loading data from HDD to memory: %.2fs\n" % (end - start))

eva_pack_size = int(query_eva_data.shape[0] / BS)

if eva_pack_size == 0:
    print("either train data rows or test data rows are less than 1024")
    sys.exit(-1)

if query_pos_data != None:
    pos_pack_size = int(query_pos_data.shape[0] / BS)
    print("Find positive data")
    print("pos_pack_size: {0}".format(pos_pack_size))
else:
    pos_pack_size = 0

print("eva_pack_size: {0}".format(eva_pack_size))

def variable_summaries(var, name, key):
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

def pull_batch(query_data, doc_data, batch_idx):
    query_in = query_data[batch_idx * BS:(batch_idx + 1) * BS, :]
    doc_in = doc_data[batch_idx * BS:(batch_idx + 1) * BS, :]
    
    query_in = query_in.toarray()
    doc_in = doc_in.toarray()

    idx_query_in = np.where(query_in > 0)
    idx_doc_in = np.where(doc_in > 0)

    """
    print(query_in.shape, doc_in.shape)
    print(np.vstack(idx_query_in).shape, query_in[idx_query_in].shape, query_in.shape)
    print(np.vstack(idx_doc_in).shape, doc_in[idx_doc_in].shape, doc_in.shape)
    """
    return (np.transpose(np.vstack(idx_query_in)), query_in[idx_query_in], query_in.shape), (np.transpose(np.vstack(idx_doc_in)), doc_in[idx_doc_in], doc_in.shape)

def feed_dict(batch_idx, is_eva = True):
    if is_eva:
        query_in, doc_in = pull_batch(query_eva_data, doc_eva_data, batch_idx)
    else:
        query_in, doc_in = pull_batch(query_pos_data, doc_pos_data, batch_idx)
    return query_in, doc_in

def append_eva_text(score_list):
    if args.evatext and args.evaappendtext and os.path.exit(args.evatext) and osp.path.exit(args.evaappendtext):
        with open(args.evatext, 'r') as eva_text_stream:
            with open(args.evaappendtext, 'w') as eva_appendtext_stream:

                idx = 0
                score_list_len = len(score_list)
                for eva_text in eva_text_stream:
                    if idx >= score_list_len:
                        eva_appendtext_stream.write("{0}\t{1}\n".format(eva_text.strip(), str(-1)))
                    else:
                        eva_appendtext_stream.write("{0}\t{1}\n".format(eva_text.strip(), score_list[idx]))
                    idx = idx + 1
                

                if idx < score_list_len:
                    print("Error: len(score_list) > len(eva_text_stream)")
                    sys.exit(-1)

config = tf.ConfigProto()  # log_device_placement=True)
config.gpu_options.allow_growth = True


tf.reset_default_graph()  
imported_meta = tf.train.import_meta_graph(os.path.join(args.modeldir, args.modelname + '.meta'))

#summary
cos_sim_list = tf.placeholder(tf.float32, shape=[None,])
variable_summaries(cos_sim_list, 'cos_sim', [COSINE_EVA_SUMMARY_KEY])
merged = tf.summary.merge_all(key=COSINE_EVA_SUMMARY_KEY)

with tf.Session(config=config) as sess:
    eva_writer = tf.summary.FileWriter(os.path.join(SUMMARIESDIR, 'eva'), sess.graph)
    pos_writer = tf.summary.FileWriter(os.path.join(SUMMARIESDIR, 'pos'), sess.graph)
    imported_meta.restore(sess, os.path.join(args.modeldir, args.modelname))

    graph = tf.get_default_graph()
    
    """
    # print all operations
    for i in graph.get_operations():
        print(i.name, i.values())
    # print all variables
    print(tf.get_default_graph().get_all_collection_keys())
    allVars = tf.global_variables()
    values = sess.run(allVars)
    for var, val in zip(allVars, values):
        print(var.name, val)
    """

    query_batch_indices = graph.get_tensor_by_name("input/QueryBatch/indices:0")
    query_batch_values = graph.get_tensor_by_name("input/QueryBatch/values:0")
    query_batch_shape = graph.get_tensor_by_name("input/QueryBatch/shape:0")
    doc_batch_indices = graph.get_tensor_by_name("input/DocBatch/indices:0")
    doc_batch_values = graph.get_tensor_by_name("input/DocBatch/values:0")
    doc_batch_shape = graph.get_tensor_by_name("input/DocBatch/shape:0")

    on_train = graph.get_tensor_by_name("input/OnTrain:0")
    
    hitprob = graph.get_tensor_by_name("Evaluate/hitprob:0")
    query_y = graph.get_tensor_by_name("L3/Relu:0")
    doc_y = graph.get_tensor_by_name("L3/Relu_1:0")

    query_norm = tf.sqrt(tf.reduce_sum(tf.square(query_y), 1, True))
    doc_norm = tf.sqrt(tf.reduce_sum(tf.square(doc_y), 1, True))
    prod = tf.reduce_sum(tf.multiply(query_y, doc_y), 1, True)
    norm_prod = tf.multiply(query_norm, doc_norm)
    cos_sim = tf.truediv(prod, norm_prod)


    eva_score_list = []
    pos_score_list = []
    #for i in range(eva_pack_size):
    for i in range(eva_pack_size):
        query_in, doc_in = feed_dict(i)
        
        score_list = sess.run(cos_sim, {query_batch_indices: query_in[0], query_batch_values: query_in[1], query_batch_shape: query_in[2],
                                   doc_batch_indices: doc_in[0], doc_batch_values: doc_in[1], doc_batch_shape: doc_in[2],
                                   on_train: False})
        
        eva_score_list = eva_score_list + [i[0] for i in score_list.tolist()]

        #print(type(score_list))
        #print(np.shape(score_list))
        #print(score_list[0:4])

    with open(args.scorelist, 'w') as score_list_stream:
        for score in eva_score_list:
            score_list_stream.write("{0}\n".format(str(score)))

    if query_pos_data != None and pos_pack_size > 0:
        for i in range(pos_pack_size):
            query_in, doc_in = feed_dict(i, False)
            
            score_list = sess.run(cos_sim, {query_batch_indices: query_in[0], query_batch_values: query_in[1], query_batch_shape: query_in[2],
                                       doc_batch_indices: doc_in[0], doc_batch_values: doc_in[1], doc_batch_shape: doc_in[2],
                                       on_train: False})

            pos_score_list = pos_score_list + [i[0] for i in score_list.tolist()]

            #print(type(score_list))
            #print(np.shape(score_list))
            #print(score_list[0:4])

    #print(np.shape(eva_score_list), eva_score_list[0:4])
    #print(np.shape(pos_score_list), pos_score_list[0:4])


    #if np.shape(pos_score_list)[0] > 1024:
    #    print(np.shape(pos_score_list), pos_score_list[0 + 1024:4 + 1024])
    #if np.shape(pos_score_list)[0] > 2048:
    #    print(np.shape(pos_score_list), pos_score_list[0 + 2048:4 + 2048])

    result = sess.run(merged, {cos_sim_list: eva_score_list})
    eva_writer.add_summary(result, 1)
    result = sess.run(merged, {cos_sim_list: pos_score_list})
    pos_writer.add_summary(result, 1)

    append_eva_text(eva_score_list)

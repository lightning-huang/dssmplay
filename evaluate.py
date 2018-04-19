import tensorflow as tf
import sys, scipy
import numpy as np

if len(sys.argv) < 5:
    print("USAGE: python evaluate.py {qnpzfile} {dnpzfile} {modelfile} {outputfile}")
    print("print the score according to original order")
    sys.exit(-1)

qnpzfile = sys.argv[1]
dnpzfile = sys.argv[2]
modelfile = sys.argv[3]
outputfile = sys.argv[4]
TRIGRAM_D = 49284
BS = 1024

query_data = scipy.sparse.load_npz(qnpzfile).tocsr()
doc_data = scipy.sparse.load_npz(dnpzfile).tocsr()
max_rows = query_data.shape[0]

def pull_batch(query_data, doc_data, max_rows, batch_idx):
    # start = time.time()
    query_in = query_data[batch_idx * BS:np.min((batch_idx + 1) * BS, max_rows), :]
    doc_in = doc_data[batch_idx * BS:np.min((batch_idx + 1) * BS, max_rows), :]

    query_in = query_in.tocoo()
    doc_in = doc_in.tocoo()

    query_in = tf.SparseTensorValue(
        np.transpose([np.array(query_in.row, dtype=np.int64), np.array(query_in.col, dtype=np.int64)]),
        np.array(query_in.data, dtype=np.float),
        np.array([BS] + query_in.shape[1:], dtype=np.int64))
    doc_in = tf.SparseTensorValue(
        np.transpose([np.array(doc_in.row, dtype=np.int64), np.array(doc_in.col, dtype=np.int64)]),
        np.array(doc_in.data, dtype=np.float),
        np.array([BS] + query_in.shape[1:], dtype=np.int64))

    # end = time.time()
    # print("Pull_batch time: %f" % (end - start))

    return query_in, doc_in


def feed_dict(query_batch, doc_batch, max_rows, batch_idx):
    query_in, doc_in = pull_batch(query_train_data, doc_train_data, max_rows, batch_idx)
    return {query_batch: query_in, doc_batch: doc_in}


sess = tf.Session()
saver = tf.train.import_meta_graph(modelfile)
saver.restore(sess,tf.train.latest_checkpoint('./'))

graph = tf.get_default_graph()
query_batch = graph.get_tensor_by_name("input/QueryBatch:0")
doc_batch = graph.get_tensor_by_name("input/DocBatch:0")
hitprob = graph.get_tensor_by_name("Evaluate/hitprob:0")

batch_size = int(np.ceil(query_data.shape[0] / BS))

for i in range(batch_size):
    sess.run(hitprob, feed_dict(query_batch, doc_batch, max_rows, i))
    flatten = hitprob.reshape(BS)
    valid_range = BS
    if (i == batch_size - 1) and (max_rows % BS != 0):
        valid_range = max_rows % BS
    with open(outputfile,'a') as scorestream:
        for j in range(valid_range):
            scorestream.write("%s\n"%flatten[j])

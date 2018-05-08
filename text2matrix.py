import pickle,sys
import numpy as np
import os,scipy
from scipy.sparse import coo_matrix

if len(sys.argv) < 4:
    print("USAGE: python text2matrix.py {datafile} {output directory} {output_prefix} [true or 1 to use 0/1 feature instead of frequencies]")
    print("for example: python text2matrix xxx.tsv d:\\data train")
    print("only process the top valid 2000*1024 pairs for both train data and test data")
    sys.exit(-1)

data_file = sys.argv[1]
output_dir = sys.argv[2]
file_prefix = sys.argv[3]

use_bool_feature = False

if len(sys.argv) >= 5:
    if sys.argv[4].lower() == 'true' or sys.argv[3] == '1':
        use_bool_feature = True
    

def build_voc_dict():
    voc_dict = {}
    alphas = 'abcdefghijklmnopqrstuvwxyz'
    digits = '0123456789'
    pound = '#'
    alphadigit = alphas + digits
    idx = 0
    # 36 ^ 3
    for c1 in alphadigit:
        for c2 in alphadigit:
            for c3 in alphadigit:
                voc_dict[''.join([c1, c2, c3])] = idx
                idx += 1
    # 36 ^ 2 * 2
    for c1 in alphadigit:
        for c2 in alphadigit:
            voc_dict[''.join(['#', c1, c2])] = idx
            idx += 1
            voc_dict[''.join([c1, c2, '#'])] = idx
            idx += 1
    # 36
    for c in alphadigit:
        voc_dict[''.join(['#', c, '#'])] = idx
        idx += 1
    print('vocabulary size=%s'%idx)
    return voc_dict


def clean_up(raw_string):
    raw_string = raw_string.lower().strip()
    outputbuffer = []
    last_c_isspace = False
    for c in raw_string:
        if c.isdigit():
            outputbuffer.append(c)
            last_c_isspace = False
        elif c.isalpha():
            outputbuffer.append(c)
            last_c_isspace = False
        else:
            #everything else treat as space
            if not last_c_isspace:
                outputbuffer.append(' ')
            last_c_isspace = True
    return ''.join(outputbuffer)

def term_stream_to_idx_array(voc_dict, term_stream):
    tokens = term_stream.split(' ')
    result_dict = {}
    for token in tokens:
        if token.strip():
            token = ''.join(['#', token, '#'])
            for i in range(len(token) - 2):
                idx = voc_dict[token[i : i+3]]
                if idx in result_dict:
                    result_dict[idx] += 1
                else:
                    result_dict[idx] = 1
    return result_dict



def write_matrix(filename, qfile, dfile, voc_dict):
    with open(filename, 'r', encoding="utf8") as pair_stream:
        read_line = 0
        feature_rows = 0
        qrow = []
        qcolumn = []
        qdata = []
        
        drow = []
        dcolumn = []
        ddata = []
        for line in pair_stream:
            read_line += 1
            line = line.strip()
            if not line:
                print("drop empty line , LN:%s"%read_line)
                continue
            tokens = line.split('\t')
            if len(tokens) < 2:
                print("drop column < 2 line, LN:%s"%read_line)
                continue
            qtext = tokens[0]
            dtext = tokens[1]
            qtext_clean = clean_up(qtext)
            dtext_clean = clean_up(dtext)
            if (not qtext_clean) or (not dtext_clean):
                print("drop empty cleaned q or d, LN:%s"%read_line)
                continue
            qdict = term_stream_to_idx_array(voc_dict, qtext_clean)
            ddict = term_stream_to_idx_array(voc_dict, dtext_clean)
            if len(qdict) <= 0 or len(ddict) <= 0:
                print("drop empty feature line for q or d, LN:%s"%read_line)
                continue
            for k in qdict:
                qrow.append(feature_rows)
                qcolumn.append(k)
                if use_bool_feature:
                    qdata.append(1)
                else:
                    qdata.append(qdict[k])
            for k in ddict:
                drow.append(feature_rows)
                dcolumn.append(k)
                if use_bool_feature:
                    ddata.append(1)
                else:
                    ddata.append(ddict[k])
            feature_rows += 1
        print("file:%s scanned %s rows, featurized %s rows"%(filename, read_line, feature_rows))
        qmatrix = coo_matrix((qdata, (qrow, qcolumn)), shape=(feature_rows, len(voc_dict)))
        dmatrix = coo_matrix((ddata, (drow, dcolumn)), shape=(feature_rows, len(voc_dict)))
        scipy.sparse.save_npz(qfile, qmatrix)
        scipy.sparse.save_npz(dfile, dmatrix)

voc_dict = build_voc_dict()
q_npz_file = os.path.join(output_dir, "query." + file_prefix + ".npz")
d_npz_file = os.path.join(output_dir, "doc." + file_prefix + ".npz")

write_matrix(data_file, q_npz_file, d_npz_file, voc_dict)

import pickle,sys
import numpy as np
from scipy.sparse import coo_matrix

if len(sys.argv) < 3:
    print("USAGE: python text2matrix.py {traindatafile} {testdatafile}")
    print("only process the top valid 2000*1024 pairs for both train data and test data")
    sys.exit(-1)

train_data_file = sys.argv[1]
test_data_file = sys.argv[2]

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
    with open(filename,'r') as pair_stream:
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
            if feature_rows >= 2000*1024:
                break
            for k in qdict:
                qrow.append(feature_rows)
                qcolumn.append(k)
                qdata.append(qdict[k])
            for k in ddict:
                drow.append(feature_rows)
                dcolumn.append(k)
                ddata.append(ddict[k])
            feature_rows += 1
        qmatrix = coo_matrix((qdata, (qrow, qcolumn)), shape=(feature_rows, len(voc_dict)))
        dmatrix = coo_matrix((ddata, (drow, dcolumn)), shape=(feature_rows, len(voc_dict)))
        with open(qfile,'wb') as f:
            pickle.dump(qmatrix,f)
        with open(dfile,'wb') as f:
            pickle.dump(dmatrix,f)

voc_dict = build_voc_dict()
write_matrix(train_data_file, "query.train.pickle", "doc.train.pickle", voc_dict)
write_matrix(test_data_file, "query.test.pickle", "doc.test.pickle", voc_dict)
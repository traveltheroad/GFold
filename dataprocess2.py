#数据处理
import os
import subprocess
import collections
import pickle as cPickle
import random
import sys
from sklearn.model_selection import train_test_split
import numpy as np
import torch
def one_hot(seq):
    RNN_seq = seq
    BASES = 'AUCG'
    bases = np.array([base for base in BASES])
    feat = np.concatenate(
        [[(bases == base.upper()).astype(int)] if str(base).upper() in BASES else np.array([[0] * len(BASES)]) for base
         in RNN_seq])

    return feat

def add_position_encoding(one_hot_matrix):
    seq_len = one_hot_matrix.shape[0]
    position_encoding = np.arange(seq_len) / (seq_len - 1)  # 生成一个从0到1的序列，长度与碱基序列相同
    position_encoding = position_encoding.reshape(-1, 1)  # 将位置编码转换为列向量
    features = np.concatenate((one_hot_matrix, position_encoding), axis=1)
    return features


def generate_label_matrix(pair_dict_all_list, seq_len):
    label_matrix = np.zeros((seq_len, seq_len), dtype=int)
    for pair in pair_dict_all_list:
        i, j = pair
        label_matrix[i, j] = 1
        label_matrix[j, i] = 1
    return label_matrix


def adjacency_to_edge_index(adjacency_matrix):
    edge_index = np.transpose(np.nonzero(adjacency_matrix))
    edge_index = torch.from_numpy(edge_index).t().contiguous()
    return edge_index


def process_bpseq_file(file_path):
    try:
        t0 = subprocess.getstatusoutput('awk \'{print $2}\' ' + file_path)
        seq = ''.join(t0[1].split('\n'))

        # 转换成one_hot
        if t0[0] == 0:
            try:
                one_hot_matrix = one_hot(seq.upper())
            except:
                raise Exception("Error occurred while converting sequence to one-hot: ", file_path)
                
        #增加位置特征后
        feature = add_position_encoding(one_hot_matrix)

        # 配对信息
        t1 = subprocess.getstatusoutput('awk \'{print $1}\' ' + file_path)
        t2 = subprocess.getstatusoutput('awk \'{print $3}\' ' + file_path)
        if t1[0] == 0 and t2[0] == 0:
            pair_dict_all_list = [[int(item_tmp) - 1, int(t2[1].split('\n')[index_tmp]) - 1] for index_tmp, item_tmp in
                                  enumerate(t1[1].split('\n')) if int(t2[1].split('\n')[index_tmp]) != 0]
        else:
            pair_dict_all_list = []
            
        #序列名字
        seq_name = os.path.basename(file_path)
        
        #序列长度
        seq_len = len(seq)
        
        #训练标签，邻接矩阵
        label_matrix= generate_label_matrix(pair_dict_all_list, seq_len)
        
        #假的邻接矩阵
        adjacency_matrix = np.zeros((seq_len, seq_len), dtype=int)
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                if (seq[i] == 'A' and seq[j] == 'U') or (seq[i] == 'U' and seq[j] == 'A') \
                        or (seq[i] == 'G' and seq[j] == 'C') or (seq[i] == 'C' and seq[j] == 'G') \
                        or (seq[i] == 'G' and seq[j] == 'U') or (seq[i] == 'U' and seq[j] == 'G'):
                    if j - i > 3:
                        adjacency_matrix[i, j] = 1
                        adjacency_matrix[j, i] = 1
                        
        #边索引
        edge_index = adjacency_to_edge_index(adjacency_matrix)
        
        #边特征
        edge_type = np.zeros((edge_index.shape[1], 3), dtype=int)
        for i in range(edge_index.shape[1]):
            if (seq[edge_index[0][i]] == 'A' and seq[edge_index[1][i]] == 'U') or (seq[edge_index[0][i]] == 'U' and seq[edge_index[1][i]] == 'A'):
                edge_type[i, :] = [1,0,0]
            elif (seq[edge_index[0][i]] == 'G' and seq[edge_index[1][i]] == 'C') or (seq[edge_index[0][i]] == 'C' and seq[edge_index[1][i]] == 'G'):
                edge_type[i, :] = [0,1,0]
            elif (seq[edge_index[0][i]] == 'G' and seq[edge_index[1][i]] == 'U') or (seq[edge_index[0][i]] == 'U' and seq[edge_index[1][i]] == 'G'):
                edge_type[i, :] = [0,0,1]
            else:
                print('有错误,{}'.format(seq_name))
        

        sample_tmp = RNA_SS_data(seq=seq, name=seq_name , length=seq_len, one_hot_matrix=one_hot_matrix, feature=feature,label_matrix=label_matrix,edge_index=edge_index,edge_type=edge_type)

        return sample_tmp

    except Exception as e:
        raise Exception("Error occurred while processing file: ", file_path, "Error message: ", str(e))

def process_directory(directory):
    all_files_list = []

    if directory.endswith(".bpseq") and not directory.startswith(".") and not directory.endswith("-checkpoint.bpseq"):
        file_path = os.path.join(root, file)
        try:
            sample_tmp = process_bpseq_file(file_path)
            all_files_list.append(sample_tmp)
        except Exception as e:
            print("Error occurred while processing file: ", file_path)
            print("Error message: ", str(e))

    return all_files_list


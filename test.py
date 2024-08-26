import sys
sys.path.append('/hy-tmp/GAT/')
from GATmodel4 import GAT
import os
import subprocess
import collections
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils import data
from postprocess_new import postprocess
from sklearn.model_selection import train_test_split
import pandas as pd

def getcsv(name,tp,fp,fn,r,p,f):
    df = pd.DataFrame({'names': name, 'tp': tp, 'fp': fp, 'fn': fn, 'recall': r, 'precision': p, 'f1': f})
    df.to_csv("archiveII_test.csv",index=False)
    return
    
def getmatch(map_no_train,seq_lens):
    dict = {}
    for i in range(seq_lens):
        if (map_no_train[i].sum() ==1):
            for j in range(seq_lens):
                if(map_no_train[i][j] == 1):
                    dict[i + 1] = j + 1
        else:
            dict[i + 1] = 0
    return dict
    
def getbpseq_new(dict,true_seq,seq_name,seq_lens):
    for i in range(seq_lens):
        list1 = [j + 1 for j in range(seq_lens)]
        list2 = []
        for j in range(seq_lens):
            list2.append(dict[j+1])
        list3 = []
        for j in true_seq:
            list3.append(j)
        download = '/hy-tmp/GAT/arch0/' + seq_name

        f = open(download, 'w')
        for k in range(0, seq_lens):
            f.write(str(list1[k]))
            f.write('   ')
            f.write(str(list3[k]))
            f.write('   ')
            f.write(str(list2[k]))
            f.write('\n')
        f.close()
        
def get_seq_new(seq_ori):
    seq = ''
    bases = 'AUCG'
    for i in seq_ori:
        base = ''
        for j, k in enumerate(i):
            if k == 1:
                base = bases[j]
                break
        if base:
            seq += base
        else:
            seq += 'N'
    return seq

def get_seq(seq_ori):
    seq = ''
    for i in seq_ori:
        for j,k in enumerate(i):
            if k == 1 and j == 0:
                seq += 'A'
            elif k == 1 and j ==1:
                seq += 'U'
            elif k == 1 and j ==2:
                seq += 'C'
            elif k == 1 and j ==3:
                seq += 'G'
            elif i.sum() == 0:
                seq += 'N'                
    return seq



def get_cut_len(data_len,set_len):
    l = data_len
    if l <= set_len:
        l = set_len
    else:
        l = (((l - 1) // 16) + 1) * 16
    return l

def evaluate_exact_new(pred_a, true_a, eps=1e-11):
    tp_map = torch.sign(torch.Tensor(pred_a)*torch.Tensor(true_a))
    tp = tp_map.sum()  #1 1
    pred_p = torch.sign(torch.Tensor(pred_a)).sum()
    true_p = true_a.sum()
    fp = pred_p - tp  #1  0
    fn = true_p - tp  #
    recall = (tp + eps)/(tp+fn+eps)
    precision = (tp + eps)/(tp+fp+eps)
    f1_score = (2*tp + eps)/(2*tp + fp + fn + eps)

    return tp,fp,fn,precision, recall, f1_score

def dice_loss(output, target):
    smooth = 1e-5

    intersection = torch.sum(output * target)
    union = torch.sum(output) + torch.sum(target)

    dice = (2. * intersection + smooth) / (union + smooth)
    dice_loss = 1. - dice

    return dice_loss

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

    if directory.endswith(".bpseq") and not directory.startswith(".") and not directory.endswith("-checkpoint.bpseq"):
        try:
            sample_tmp = process_bpseq_file(directory)
        except Exception as e:
            print("Error occurred while processing file: ", directory)
            print("Error message: ", str(e))

    return sample_tmp
    
def get_all_files_in_directory(directory_path):
    # 用于存储所有文件路径的列表
    file_paths = []

    # 遍历目录下的所有文件和子目录
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            # 使用os.path.join()构建文件的绝对路径
            file_path = os.path.join(root, file)
            file_paths.append(file_path)

    return file_paths

test_path = '/hy-tmp/archiveII//'
test_files = get_all_files_in_directory(test_path)
print(len(test_files))

def f1_loss(f1):
    loss = 1 / f1
    return loss

def dice_loss(output, target):
    smooth = 1e-5

    intersection = torch.sum(output * target)
    union = torch.sum(output) + torch.sum(target)

    dice = (2. * intersection + smooth) / (union + smooth)
    dice_loss = 1. - dice

    return dice_loss

error = []
torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GAT(in_channels=5, hidden_channels=16, out_channels=16, num_heads=6)
model.load_state_dict(torch.load('/hy-tmp//RNAS930.pt',map_location='cuda:0'))
model.to(device)
print(model)
pos_weight = torch.Tensor([600]).to(device)
criterion_bce_weighted = torch.nn.BCEWithLogitsLoss(pos_weight = pos_weight)
optimizer = optim.Adam(model.parameters())
optimizer.zero_grad()
best_f1 = 0.0
best_epoch = -1
RNA_SS_data = collections.namedtuple('RNA_SS_data','seq name length one_hot_matrix feature label_matrix edge_index edge_type')
i=0
test_list = []
for test_file in test_files:
    i = i + 1
    print(i)
    try:
        rna = process_directory(test_file)
        test_list.append(rna)
    except Exception as e:
        print("发生了异常：", e)
        continue
print(len(test_list))
    
for epoch in range(1):
    
    step = 0
    step1 = 0
    loss_all = 0
    f1 = 0
    p1 = 0
    r1 = 0
    k = 0
    seq_names = []
    tp_list = []
    fp_list = []
    fn_list = []
    recall = []
    precision = []
    f1_score =[]

    for rna in test_list:
        results_df = pd.DataFrame(columns=['name', 'tp', 'fp', 'fn','f1','precision','recall'])
        model.eval()        
        feature = torch.Tensor(rna.feature.astype(float)).to(device)
        feature = torch.squeeze(feature, dim=0)

        l = get_cut_len(feature.shape[0],80)
        edge_index = torch.Tensor(rna.edge_index).long().to(device)
        edge_index = torch.squeeze(edge_index, dim=0)
        edge_type = torch.Tensor(rna.edge_type.astype(float)).to(device)
        edge_type = torch.squeeze(edge_type, dim=0)
        label = torch.Tensor(rna.label_matrix.astype(float)).to(device)
        label = label.unsqueeze(0)
        step1 =step1 + 1
        
        with torch.no_grad():
            predict = model(feature, edge_index,edge_type,l)
        predict = predict[:,:feature.shape[0],:feature.shape[0]]
        feature = feature[:, :4]
        feature = torch.unsqueeze(feature, dim=0)
        predict_no_train = postprocess(predict,
            feature, 0.01, 0.1, 100, 1.6, True,1.5)
        map_no_train = (predict_no_train > 0.5).float()

        tp,fp,fn,p,r,f = evaluate_exact_new(map_no_train,label)
        f1 = f + f1
        p1 = p + p1
        r1 = r + r1
        
        seq_names.append(rna.name)
        tp_list.append(tp.cpu().item())
        fp_list.append(fp.cpu().item())
        fn_list.append(fn.cpu().item())
        recall.append(r.cpu().item())
        precision.append(p.cpu().item())
        f1_score.append(f.cpu().item())
     
        dict = getmatch(map_no_train[0],label.shape[1])
        true_seq = get_seq_new(feature[0])
        try:
            getbpseq_new(dict,true_seq,rna.name,label.shape[1])
        except:
            print('=====')
            error.append(name)
            print(name)
        i = i + 1
        print(i)
    getcsv(seq_names,tp_list,fp_list,fn_list,recall,precision,f1_score)

    print('val_f1: {}, val_p1: {}, val_r1: {}'.format(f1/step1, p1/step1,r1/step1))


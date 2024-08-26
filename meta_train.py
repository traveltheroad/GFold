import sys
sys.path.append('/mnt/workspace//GAT_Train/')
from GATmodel4 import GAT
from datagenerate import RNADataProcessor,Dataset,RNADataProcessor1
import os
import subprocess
import collections
import pickle as cPickle
import random
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from postprocess_new import postprocess
import subprocess
import random
from copy import deepcopy

def task_sample(folder_path,num_rna):
    train_data_list = []
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.cPickle')]
    # 随机选择一个文件路径
    task_file_path = random.choice(file_paths)
    print("random choice task {}".format(task_file_path))
    # 加载文件中的数据作为任务
    with open(task_file_path, 'rb') as f:
        task = cPickle.load(f,encoding='iso-8859-1')
    if len(task)>100:
        inner = 8
        task = random.sample(task, num_rna)
        train_data_list.append(RNADataProcessor1(task))
        return inner,train_data_list
    else:
        train_data_list.append(RNADataProcessor1(task))
        inner = 20
        return inner,train_data_list
    


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
    return precision, recall, f1_score



# RNA_SS_data = collections.namedtuple('RNA_SS_data','seq name length one_hot_matrix feature label_matrix edge_index edge_type')
# # 定义要遍历的目录
path = "/mnt/workspace/RNAStralign_dataset/train/"
path1 = "/mnt/workspace/RNAStralign_dataset/test/"


class learning:
    def __init__(self, model, outer_step_size=0.1, outer_iterations=3000, inner_iterations=8):
        self.model = model
        self.outer_step_size = outer_step_size
        self.outer_iterations = outer_iterations
        self.inner_iterations = inner_iterations
        self.best_f1 = 0
    def reset(self):
        self.model.zero_grad()
        self.current_loss = 0
        self.current_batch = 0
        
    def predict(self, x):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")       
        step1 = 0
        f1 = 0
        p1 = 0
        r1 = 0
        for feature, edge_index, edge_type,label,name in x:
            (self.model).eval()
            feature = torch.Tensor(feature.float()).to(device)
            feature = torch.squeeze(feature, dim=0)
            l = get_cut_len(feature.shape[0],80)
            edge_index = torch.Tensor(edge_index).long().to(device)
            edge_index = torch.squeeze(edge_index, dim=0)
            edge_type = torch.Tensor(edge_type.float()).to(device)
            edge_type = torch.squeeze(edge_type, dim=0)
            label = torch.Tensor(label.float()).to(device)
            step1 =step1 + 1

            with torch.no_grad():
                predict = self.model(feature, edge_index,edge_type,l)
            predict = predict[:,:feature.shape[0],:feature.shape[0]]
            feature = feature[:, :4]
            feature = torch.unsqueeze(feature, dim=0)
            predict_no_train = postprocess(predict,
                feature, 0.01, 0.1, 100, 1.6, True,1.5)
            map_no_train = (predict_no_train > 0.5).float()
            p,r,f = evaluate_exact_new(map_no_train,label)
            f1 = f + f1
            p1 = p + p1
            r1 = r + r1
        return f1/step1, p1/step1, r1/step1
    
    def train(self):
        RNA_SS_data = collections.namedtuple('RNA_SS_data','seq name length one_hot_matrix feature label_matrix edge_index edge_type')
        
        params = {'batch_size': 1,
                'shuffle': True,
                'num_workers': 6,
                'drop_last': True}
        test_data_list=[]
        path1 = "/mnt/workspace/RNAStralign_dataset/test/"
        # 遍历目录下所有文件夹及文件
        for root, dirs, files in os.walk(path1):
            for file in files:
                print(file)
                test_data_list.append(RNADataProcessor(os.path.join(root, file)))
        test_set = Dataset(test_data_list)
        test_generator = data.DataLoader(test_set, **params)
        num_batches = len(test_generator)
        print("DataLoader对象中的批次数量：", num_batches)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pos_weight = torch.Tensor([600]).to(device)
        criterion_bce_weighted = torch.nn.BCEWithLogitsLoss(pos_weight = pos_weight)
        optimizer = optim.Adam((self.model).parameters())
        self.reset()
        for outer_iteration in range(self.outer_iterations):
            self.reset()
            inner1,datalist = task_sample("/mnt/workspace/RNAStralign_dataset/train/", 100)
            train_merge = Dataset(datalist)
            train_generator = data.DataLoader(train_merge, **params)
            num_batches = len(train_generator)
            print("DataLoader对象中的批次数量：", num_batches)
            current_weights = deepcopy(self.model.state_dict())
            #内循环
            for inner_iteration in range(inner1):
                for feature, edge_index, edge_type,label,name in train_generator:
                    self.model.train()
                    feature = torch.Tensor(feature.float()).to(device)
                    feature = torch.squeeze(feature, dim=0)
                    l = get_cut_len(feature.shape[0],80)
                    edge_index = torch.Tensor(edge_index).long().to(device)
                    edge_index = torch.squeeze(edge_index, dim=0)
                    edge_type = torch.Tensor(edge_type.float()).to(device)
                    edge_type = torch.squeeze(edge_type, dim=0)
                    label = torch.Tensor(label.float()).to(device)

                    predict = self.model(feature, edge_index,edge_type,l)
                    label2 = torch.zeros_like(predict).to(device)
                    label2[:,:label.shape[1],:label.shape[1]] = label
                    loss_a = criterion_bce_weighted(predict, label2)
                    optimizer.zero_grad()
                    loss_a.backward()
                    optimizer.step()
                    loss_tensor = loss_a.cpu()
                    self.current_loss += loss_tensor.data.numpy()
                    self.current_batch += 1
            #外循环参数更新        
            candidate_weights = self.model.state_dict()
            alpha = self.outer_step_size * (1 - outer_iteration / self.outer_iterations)
            state_dict = {candidate: (current_weights[candidate] + alpha * 
                        (candidate_weights[candidate] - current_weights[candidate])) 
                        for candidate in candidate_weights}
            self.model.load_state_dict(state_dict)
            
            print("Iteration {}: , current_batch = {} ,train_Loss = {}".format(outer_iteration, self.current_batch, self.current_loss / self.current_batch))

 
            # 在测试集上评估泛化性
            f1, p, r = self.predict(test_generator)
            if f1 > self.best_f1:
                self.best_f1 = f1
                torch.save(self.model.state_dict(), "/mnt/workspace/best_model_meta11.pt")

            print("test ====== Iteration {}:,f1 = {},  p = {},  r = {},".format(outer_iteration, f1,p,r))























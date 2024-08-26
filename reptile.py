import sys
sys.path.append('/mnt/workspace//GAT_Train/')
import torch 
import torch.nn as nn
import random
import math
from time import time
from copy import deepcopy
from datetime import datetime
import pickle as cPickle
from torch.utils import data
import torch.optim as optim
import os
import subprocess
import random
from pytorch_model import U_Net as Net
from meta_learning.utils import *
from meta_learning.data_generator import RNASSDataGenerator, Dataset,Dataset_merge_two
import collections
from sklearn.model_selection import train_test_split
from meta_learning.postprocess import postprocess


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
        task = random.sample(task, num_rna)
        train_data_list.append(RNASSDataGenerator(task))
        return train_data_list
    else:
        train_data_list.append(RNASSDataGenerator(task))
        return train_data_list
    
def test_sample(folder_path):
    test_data_list = []
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.cPickle')]
    for i in file_paths:
        with open(i, 'rb') as f:
            task = cPickle.load(f,encoding='iso-8859-1')
            test_data_list.append(RNASSDataGenerator(task))
    return test_data_list

class learning:
    def __init__(self, model, outer_step_size=0.1, outer_iterations=1000, inner_iterations=8):
        self.model = model
        self.outer_step_size = outer_step_size
        self.outer_iterations = outer_iterations
        self.inner_iterations = inner_iterations
    def reset(self):
        self.model.zero_grad()
        self.current_loss = 0
        self.current_batch = 0
        self.val_loss = 0
        self.val_batch = 0
        self.test_loss = 0
        self.test_batch = 0
        
    def predict(self, x):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        result_no_train = list()
        pos_weight = torch.Tensor([300]).to(device)
        criterion_bce_weighted = torch.nn.BCEWithLogitsLoss(
            pos_weight = pos_weight)
        i = 0
        loss = 0
        for contacts, seq_embeddings, seq_lens, seq_ori, seq_name in x: 
            contacts_batch = torch.Tensor(contacts.float()).to(device)
            seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)
            seq_ori_batch = torch.Tensor(seq_ori.float()).to(device)
            with torch.no_grad():
                pred_contacts = self.model(seq_embedding_batch)
            contact_masks = torch.zeros_like(pred_contacts)
            contact_masks[:, :seq_lens, :seq_lens] = 1
            # Compute loss
            loss_u = criterion_bce_weighted(pred_contacts*contact_masks, contacts_batch)
            u_no_train = postprocess(pred_contacts,
                seq_ori_batch, 0.01, 0.1, 100, 1.6, True,1.5)
            map_no_train = (u_no_train > 0.5).float()

            result_no_train_tmp = list(map(lambda i: evaluate_exact_new(map_no_train.cpu()[i],
                contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
            result_no_train += result_no_train_tmp
            i = i + 1             
            loss = loss + loss_u

        nt_exact_p,nt_exact_r,nt_exact_f1 = zip(*result_no_train)

        print('Average testing F1 score with pure post-processing: ', np.average(nt_exact_f1))
        print('Average testing precision with pure post-processing: ', np.average(nt_exact_p))
        print('Average testing recall with pure post-processing: ', np.average(nt_exact_r))
        print('Average loss:', loss / i)

        return i, loss , np.average(nt_exact_f1), np.average(nt_exact_p), np.average(nt_exact_r)

    def train(self):
        RNA_SS_data = collections.namedtuple('RNA_SS_data','seq ss_label length name pairs')
        params = {'batch_size': 1,
                'shuffle': True,
                'num_workers': 6,
                'drop_last': True}

        test_merge = Dataset_merge_two(test_sample('/hy-tmp/meta_learning_rna/rna_data_cross_family/'))
        test_merge_generator = data.DataLoader(test_merge, **params)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pos_weight = torch.Tensor([300]).to(device)
        criterion_bce_weighted = torch.nn.BCEWithLogitsLoss(pos_weight = pos_weight)
        u_optimizer = optim.Adam((self.model).parameters())
        self.reset()
        for outer_iteration in range(self.outer_iterations):
            train_merge = Dataset(task_sample('/hy-tmp/meta_learning_rna/rna_data/', 100))
            train_data, val_data = train_test_split(train_merge, test_size=0.2, random_state=42)
            train_merge_generator = data.DataLoader(train_data, **params)
            val_merge_generator = data.DataLoader(val_data, **params)
            current_weights = deepcopy(self.model.state_dict())
            loss = 0
            #内循环
            for inner_iteration in range(self.inner_iterations):
                for contacts, seq_embeddings, seq_lens, seq_ori, seq_name in train_merge_generator:

                    contacts_batch = torch.Tensor(contacts.float()).to(device)
                    seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)
                    pred_contacts = self.model(seq_embedding_batch)
                    contact_masks = torch.zeros_like(pred_contacts)
                    contact_masks[:, :seq_lens, :seq_lens] = 1
                    loss_u = criterion_bce_weighted(pred_contacts*contact_masks, contacts_batch)
                    u_optimizer.zero_grad()
                    loss_u.backward()
                    u_optimizer.step()
                    loss_tensor = loss_u.cpu()
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

            # 在验证集上评估模型性能
            torch.cuda.empty_cache()
            val_batch , loss, f1, p, r = self.predict(val_merge_generator)
            self.val_loss += loss
            self.val_batch += val_batch
            print("Iteration {}:,val_Loss = {}".format(outer_iteration, self.val_loss / self.val_batch))
            # 在测试集上评估泛化性
            test_batch , loss, f1, p, r = self.predict(test_merge_generator)
            self.test_loss += loss
            self.test_batch += test_batch
            print("Iteration {}:,test_Loss = {}".format(outer_iteration, self.test_loss / self.test_batch))


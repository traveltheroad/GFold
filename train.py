import sys
sys.path.append('/mnt/workspace//GAT/')
from GATmodel4 import GAT
from datagenerate import RNADataProcessor,Dataset
import os
import subprocess
import collections
import pickle as cPickle
import random
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.optim as optim
from torch.utils import data
from postprocess_new import postprocess
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



RNA_SS_data = collections.namedtuple('RNA_SS_data','seq name length one_hot_matrix feature label_matrix edge_index edge_type')
params = {'batch_size': 1,
          'shuffle': True,
          'num_workers': 6,
          'drop_last': True}
train_data_list = []
test_data_list = []
# 定义要遍历的目录
path = "/mnt/workspace//1/"

# 遍历目录下所有文件夹及文件
for root, dirs, files in os.walk(path):
    for file in files:
        print(file)
        # 输出文件路径
        train_data_list.append(RNADataProcessor(os.path.join(root, file)))

train_set = Dataset(train_data_list)
train_generator = data.DataLoader(train_set, **params)

num_batches = len(train_generator)
print("DataLoader对象中的批次数量：", num_batches)

path1 = "/mnt/workspace/2/"

# 遍历目录下所有文件夹及文件
for root, dirs, files in os.walk(path1):
    for file in files:
        print(file)
        # 输出文件路径
        test_data_list.append(RNADataProcessor(os.path.join(root, file)))
test_set = Dataset(test_data_list)
test_generator = data.DataLoader(test_set, **params)

num_batches = len(test_generator)
print("DataLoader对象中的批次数量：", num_batches)

torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GAT(in_channels=5, hidden_channels=16, out_channels=16, num_heads=6)
# model.load_state_dict(torch.load('/mnt/workspace/best_mode.pt',map_location='cuda:0'))
#model.load_state_dict(torch.load('/mnt/workspace/gatmodel_train_125.pt',map_location='cuda:0'))
model.to(device)
print(model)
pos_weight = torch.Tensor([600]).to(device)
criterion_bce_weighted = torch.nn.BCEWithLogitsLoss(pos_weight = pos_weight)
optimizer = optim.Adam(model.parameters())
optimizer.zero_grad()
best_f1 = 0.0
best_epoch = -1

for epoch in range(2000):
    
    step = 0
    step1 = 0
    loss_all = 0
    f1 = 0
    p1 = 0
    r1 = 0
    for feature, edge_index, edge_type,label,name in train_generator:

        model.train()
        feature = torch.Tensor(feature.float()).to(device)
        feature = torch.squeeze(feature, dim=0)
        l = get_cut_len(feature.shape[0],80)
        edge_index = torch.Tensor(edge_index).long().to(device)
        edge_index = torch.squeeze(edge_index, dim=0)
        edge_type = torch.Tensor(edge_type.float()).to(device)
        edge_type = torch.squeeze(edge_type, dim=0)
        label = torch.Tensor(label.float()).to(device)
        
        predict = model(feature, edge_index,edge_type,l)
        predict = predict[:,:feature.shape[0],:feature.shape[0]]
        # loss_a = criterion_bce_weighted(predict, label)   
        # feature = feature[:, :4]
        # feature = torch.unsqueeze(feature, dim=0)
        # predict_no_train = postprocess(predict,
        #      feature, 0.01, 0.1, 100, 1.6, True,1.5)
        # map_no_train = (predict_no_train > 0.5).float()
        #loss_b = dice_loss(predict,label)
        #loss = loss_a +  loss_b
        # optimizer.zero_grad()
        #loss.backward()
        # loss_a.backward()
        # optimizer.step()
        #loss_all = loss.item() + loss_all
        # loss_all = loss_a.item() + loss_all
        # step = step + 1
        loss_a = criterion_bce_weighted(predict, label)
        optimizer.zero_grad()
        loss_a.backward()
        optimizer.step()
        loss_all = loss_a.item() + loss_all
        step = step + 1
        

    print('Training, epoch {}, step: {}, train_loss_average: {}'.format(epoch,step, loss_all/step))

#     for feature, edge_index, edge_type,label,name in test_generator:
#         model.eval()
#         feature = torch.Tensor(feature.float()).to(device)
#         feature = torch.squeeze(feature, dim=0)
#         l = get_cut_len(feature.shape[0],80)
#         edge_index = torch.Tensor(edge_index).long().to(device)
#         edge_index = torch.squeeze(edge_index, dim=0)
#         edge_type = torch.Tensor(edge_type.float()).to(device)
#         edge_type = torch.squeeze(edge_type, dim=0)
#         label = torch.Tensor(label.float()).to(device)
#         step1 =step1 + 1
        
#         with torch.no_grad():
#             predict = model(feature, edge_index,edge_type,l)
#         predict = predict[:,:feature.shape[0],:feature.shape[0]]
#         feature = feature[:, :4]
#         feature = torch.unsqueeze(feature, dim=0)
#         predict_no_train = postprocess(predict,
#             feature, 0.01, 0.1, 100, 1.6, True,1.5)
#         map_no_train = (predict_no_train > 0.5).float()       
  
#         p,r,f = evaluate_exact_new(map_no_train,label)
#         f1 = f + f1
#         p1 = p + p1
#         r1 = r + r1
        

            
#     if (f1/step1) > best_f1:
#         best_f1 = (f1/step1)
#         best_epoch = epoch
#         torch.save(model.state_dict(), "/mnt/workspace/best_model_next2.pt")
#         print(('best Epoch {}, best train_loss: {}, best val_f1: {}, best val_p1: {}, best val_r1: {}'.format(epoch, loss_a/step, f1/step1, p1/step1,r1/step1)))
#     print('Epoch {}, train_loss: {}, val_f1: {}, val_p1: {}, val_r1: {}'.format(epoch, loss_a/step, f1/step1, p1/step1,r1/step1))
#     # if epoch > 50:
    #     torch.save(model.state_dict(),  f'/mnt/workspace/gatmodel_train_{epoch}.pt')
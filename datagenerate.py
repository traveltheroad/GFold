import os
import pickle as cPickle
import collections
from multiprocessing import Pool
from torch.utils import data
from collections import Counter
from random import shuffle
import torch
import numpy as np

class RNADataProcessor1(object):
    def __init__(self, data_dir):
        self.data = data_dir 
        self.load_data()

    def load_data(self):
        p = Pool()
        # Load the current split
        RNA_SS_data = collections.namedtuple('RNA_SS_data','seq name length one_hot_matrix feature label_matrix edge_index edge_type')
        self.seq = np.array([instance[0] for instance in self.data]) 
        self.name = np.array([instance[1] for instance in self.data])  
        self.length = np.array([instance[2] for instance in self.data])  
        self.one_hot_matrix = np.array([instance[3] for instance in self.data]) 
        self.feature = np.array([instance[4] for instance in self.data])
        self.label_matrix = np.array([instance[5] for instance in self.data])
        self.edge_index = np.array([instance[6] for instance in self.data])
        self.edge_type = np.array([instance[7] for instance in self.data])
        self.len = len(self.data) 
        
    def get_one_sample(self, index):
        label = self.label_matrix[index]
        feature = self.feature[index]
        edge_index = self.edge_index[index]
        edge_type = self.edge_type[index]
        name = self.name[index]
        return feature, edge_index, edge_type, label,name






class RNADataProcessor(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir 
        self.load_data()

    def load_data(self):
        p = Pool()
        data_dir = self.data_dir
        # Load the current split
        RNA_SS_data = collections.namedtuple('RNA_SS_data','seq name length one_hot_matrix feature label_matrix edge_index edge_type')
        with open(self.data_dir, 'rb') as f:
            self.data = cPickle.load(f)

        self.seq = np.array([instance[0] for instance in self.data]) 
        self.name = np.array([instance[1] for instance in self.data])  
        self.length = np.array([instance[2] for instance in self.data])  
        self.one_hot_matrix = np.array([instance[3] for instance in self.data]) 
        self.feature = np.array([instance[4] for instance in self.data])
        self.label_matrix = np.array([instance[5] for instance in self.data])
        self.edge_index = np.array([instance[6] for instance in self.data])
        self.edge_type = np.array([instance[7] for instance in self.data])
        #self.len = len(self.data) 
        
    def get_one_sample(self, index):
        label = self.label_matrix[index]
        feature = self.feature[index]
        edge_index = self.edge_index[index]
        edge_type = self.edge_type[index]
        name = self.name[index]
        return feature, edge_index, edge_type, label,name


class Dataset(data.Dataset):

    def __init__(self, data_list):
        self.data2 = data_list[0]
        self.data2.len = len(self.data2.name)
        if len(data_list) > 1:
            self.data = self.merge_data(data_list)
        else:
            self.data = self.data2

    def __len__(self):
        return self.data.len
    
    def merge_data(self,data_list):
	
        self.data2.seq = np.concatenate((data_list[0].seq,data_list[1].seq),axis=0)
        self.data2.name = np.concatenate((data_list[0].name,data_list[1].name),axis=0)
        self.data2.length = np.concatenate((data_list[0].length,data_list[1].length),axis=0)
        self.data2.one_hot_matrix = np.concatenate((data_list[0].one_hot_matrix,data_list[1].one_hot_matrix),axis=0)
        self.data2.feature = np.concatenate((data_list[0].feature,data_list[1].feature),axis=0)
        self.data2.label_matrix = np.concatenate((data_list[0].label_matrix,data_list[1].label_matrix),axis=0)
        self.data2.edge_index = np.concatenate((data_list[0].edge_index,data_list[1].edge_index),axis=0)
        self.data2.edge_type = np.concatenate((data_list[0].edge_type,data_list[1].edge_type),axis=0)

        
        for item in data_list[2:]:
                self.data2.seq = np.concatenate((self.data2.seq, item.seq), axis=0)
                self.data2.name = np.concatenate((self.data2.name, item.name), axis=0)
                self.data2.length = np.concatenate((self.data2.length, item.length), axis=0)
                self.data2.one_hot_matrix = np.concatenate((self.data2.one_hot_matrix, item.one_hot_matrix), axis=0)
                self.data2.feature = np.concatenate((self.data2.feature, item.feature), axis=0)
                self.data2.label_matrix = np.concatenate((self.data2.label_matrix, item.label_matrix), axis=0)
                self.data2.edge_index = np.concatenate((self.data2.edge_index, item.edge_index), axis=0)
                self.data2.edge_type = np.concatenate((self.data2.edge_type, item.edge_type), axis=0)

        self.data2.len = len(self.data2.name)
        return self.data2    

    def __getitem__(self, index):
        # Select sample
        return self.data.get_one_sample(index)
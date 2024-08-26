import pickle as cPickle
import sys
sys.path.append('/mnt/workspace/GAT_Train/')
import os
import torch
import torch.optim as optim
import subprocess
import random
from GATmodel4 import GAT
from datagenerate import RNADataProcessor,Dataset,RNADataProcessor1
import collections
from sklearn.model_selection import train_test_split
from meta_train import learning
def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GAT(in_channels=5, hidden_channels=16, out_channels=16, num_heads=6)
    model.to(device)
    mlearning = learning(model)
    mlearning.train()


if __name__ == '__main__':
    """
    See module-level docstring for a description of the script.
    """
    RNA_SS_data = collections.namedtuple('RNA_SS_data','seq name length one_hot_matrix feature label_matrix edge_index edge_type')
    main()
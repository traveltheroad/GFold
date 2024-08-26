#网络模型
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch import einsum
from torch.nn import init
CH_FOLD2 = 1
torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")






class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class GAT(nn.Module):
  
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads,img_ch=32,output_ch=1):
        super(GAT, self).__init__()

        self.GATconv1 = GATConv(in_channels, hidden_channels, heads=num_heads,add_self_loops=False,edge_dim=3)
        #self.GATconv2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads,add_self_loops=False,edge_dim=3)
        #self.GATconv3 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads,add_self_loops=False,edge_dim=3)
        self.GATconv4 = GATConv(hidden_channels * num_heads, out_channels, heads=1,add_self_loops=False,edge_dim=3)
        #self.relu = nn.ReLU()
        #self.attention = Attn(dim=out_channels, query_key_dim=out_channels, expansion_factor=2., dropout=0.1)

        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Conv1 = conv_block(ch_in=img_ch,ch_out=int(32*CH_FOLD2))
        self.Conv2 = conv_block(ch_in=int(32*CH_FOLD2),ch_out=int(64*CH_FOLD2))
        self.Conv3 = conv_block(ch_in=int(64*CH_FOLD2),ch_out=int(128*CH_FOLD2))
        self.Conv4 = conv_block(ch_in=int(128*CH_FOLD2),ch_out=int(256*CH_FOLD2))
        self.Conv5 = conv_block(ch_in=int(256*CH_FOLD2),ch_out=int(512*CH_FOLD2))

        self.Up5 = up_conv(ch_in=int(512*CH_FOLD2),ch_out=int(256*CH_FOLD2))
        self.Up_conv5 = conv_block(ch_in=int(512*CH_FOLD2), ch_out=int(256*CH_FOLD2))

        self.Up4 = up_conv(ch_in=int(256*CH_FOLD2),ch_out=int(128*CH_FOLD2))
        self.Up_conv4 = conv_block(ch_in=int(256*CH_FOLD2), ch_out=int(128*CH_FOLD2))
        
        self.Up3 = up_conv(ch_in=int(128*CH_FOLD2),ch_out=int(64*CH_FOLD2))
        self.Up_conv3 = conv_block(ch_in=int(128*CH_FOLD2), ch_out=int(64*CH_FOLD2))
        
        self.Up2 = up_conv(ch_in=int(64*CH_FOLD2),ch_out=int(32*CH_FOLD2))
        self.Up_conv2 = conv_block(ch_in=int(64*CH_FOLD2), ch_out=int(32*CH_FOLD2))

        self.Conv_1x1 = nn.Conv2d(int(32*CH_FOLD2),output_ch,kernel_size=1,stride=1,padding=0)
        
    def matrix_rep(self, x):
        '''
        for each position i,j of the matrix, we concatenate the embedding of i and j
        '''
        #x = x.permute(0, 2, 1) # L*d
        L = x.shape[1]
        x2 = x
        x = x.unsqueeze(1)
        x2 = x2.unsqueeze(2)
        x = x.repeat(1, L,1,1)
        x2 = x2.repeat(1, 1, L,1)
        mat = torch.cat([x,x2],-1) # L*L*2d

        # make it symmetric
        # mat_tril = torch.cat(
        #     [torch.tril(mat[:,:, i]) for i in range(mat.shape[-1])], -1)
        mat_tril = torch.tril(mat.permute(0, -1, 1, 2)) # 2d*L*L
        mat_diag = mat_tril - torch.tril(mat.permute(0, -1, 1, 2), diagonal=-1)
        mat = mat_tril + torch.transpose(mat_tril, -2, -1) - mat_diag
        return mat

    
    def forward(self, x, edge_index,edge_attr,l):
        #embedding
        x = self.GATconv1(x, edge_index,edge_attr)
        #x = self.GATconv2(x, edge_index,edge_attr)
        #x = self.GATconv3(x, edge_index, edge_attr)
        x = self.GATconv4(x, edge_index, edge_attr)
        x = x.unsqueeze(0)
        x = self.matrix_rep(x)
        #x = torch.sigmoid(x)

        target_tensor = torch.zeros(1,32,l,l).to(device)
        target_tensor[:,:,:x.shape[2],:x.shape[2]] = x
        #training
        x1 = self.Conv1(target_tensor)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)
        
        d1 = self.Conv_1x1(d2)
        d1 = d1.squeeze(1)
        return torch.transpose(d1, -1, -2) * d1
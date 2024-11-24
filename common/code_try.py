## Our PoseFormer model was revised from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

import math
import logging
from functools import partial
from collections import OrderedDict
from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model


class HGraphConv(nn.Module):
    """
    High-order graph convolution layer 
    """

    def __init__(self, in_features, out_features, adj, bias=True):
        super(HGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(10, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
  
        self.adj_0 = torch.eye(adj.size(0), dtype=torch.float) # declare self-connections
        self.m_0 = (self.adj_0 > 0)
        self.e_0 =  nn.Parameter(torch.zeros(1, len(self.m_0.nonzero()), dtype=torch.float))
        nn.init.constant_(self.e_0.data, 1)

        self.adj_1 = adj # one_hop neighbors
        self.m_1 = (self.adj_1 > 0)
        self.e_1 = nn.Parameter(torch.zeros(1, len(self.m_1.nonzero()), dtype=torch.float))
        nn.init.constant_(self.e_1.data, 1)
        
        self.adj_2 = torch.matmul(self.adj_1, adj) # two_hop neighbors
        self.m_2 = (self.adj_2 > 0)
        self.e_2 = nn.Parameter(torch.zeros(1, len(self.m_2.nonzero()), dtype=torch.float))
        nn.init.constant_(self.e_2.data, 1)

        self.adj_3 = torch.matmul(self.adj_2, adj) # three_hop neighbors
        self.m_3 = (self.adj_3 > 0)
        self.e_3 = nn.Parameter(torch.zeros(1, len(self.m_3.nonzero()), dtype=torch.float))
        nn.init.constant_(self.e_3.data,1)

        self.adj_4 = torch.matmul(self.adj_3, adj) # three_hop neighbors
        self.m_4 = (self.adj_4 > 0)
        self.e_4 = nn.Parameter(torch.zeros(1, len(self.m_4.nonzero()), dtype=torch.float))
        nn.init.constant_(self.e_4.data,1)

        self.adj_5 = torch.matmul(self.adj_4, adj) # three_hop neighbors
        self.m_5 = (self.adj_5 > 0)
        self.e_5 = nn.Parameter(torch.zeros(1, len(self.m_5.nonzero()), dtype=torch.float))
        nn.init.constant_(self.e_5.data,1)

        self.adj_6 = torch.matmul(self.adj_5, adj) # three_hop neighbors
        self.m_6 = (self.adj_6 > 0)
        self.e_6 = nn.Parameter(torch.zeros(1, len(self.m_6.nonzero()), dtype=torch.float))
        nn.init.constant_(self.e_6.data,1)

        self.adj_7 = torch.matmul(self.adj_6, adj) # three_hop neighbors
        self.m_7 = (self.adj_7 > 0)
        self.e_7 = nn.Parameter(torch.zeros(1, len(self.m_7.nonzero()), dtype=torch.float))
        nn.init.constant_(self.e_7.data,1)

        self.adj_8 = torch.matmul(self.adj_7, adj) # three_hop neighbors
        self.m_8 = (self.adj_8 > 0)
        self.e_8 = nn.Parameter(torch.zeros(1, len(self.m_8.nonzero()), dtype=torch.float))
        nn.init.constant_(self.e_8.data,1)

        self.adj_9 = torch.matmul(self.adj_8, adj) # three_hop neighbors
        self.m_9 = (self.adj_9 > 0)
        self.e_9 = nn.Parameter(torch.zeros(1, len(self.m_9.nonzero()), dtype=torch.float))
        nn.init.constant_(self.e_9.data,1)



        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float)) #self.bias = nn.Parameter(torch.zeros(out_features*4, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
            
            self.bias_2 = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias_2.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)



    def forward(self, input):


        h0 = torch.matmul(input, self.W[0])
        h1 = torch.matmul(input, self.W[1])
        h2 = torch.matmul(input, self.W[2])
        h3 = torch.matmul(input, self.W[3])
        h4 = torch.matmul(input, self.W[4])
        h5 = torch.matmul(input, self.W[5])
        h6 = torch.matmul(input, self.W[6])
        h7 = torch.matmul(input, self.W[7])
        h8 = torch.matmul(input, self.W[8])
        h9 = torch.matmul(input, self.W[9])

    
        A_0 = -9e15 * torch.ones_like(self.adj_0).to(input.device)
        A_1 = -9e15 * torch.ones_like(self.adj_1).to(input.device) # without self-connection 
        A_2 = -9e15 * torch.ones_like(self.adj_2).to(input.device)
        A_3 = -9e15 * torch.ones_like(self.adj_3).to(input.device)
        A_4 = -9e15 * torch.ones_like(self.adj_4).to(input.device)
        A_5 = -9e15 * torch.ones_like(self.adj_5).to(input.device) # without self-connection 
        A_6 = -9e15 * torch.ones_like(self.adj_6).to(input.device)
        A_7 = -9e15 * torch.ones_like(self.adj_7).to(input.device)
        A_8 = -9e15 * torch.ones_like(self.adj_8).to(input.device) # without self-connection 
        A_9 = -9e15 * torch.ones_like(self.adj_9).to(input.device)

         
        A_0[self.m_0] = self.e_0
        A_1[self.m_1] = self.e_1
        A_2[self.m_2] = self.e_2
        A_3[self.m_3] = self.e_3
        A_4[self.m_4] = self.e_4
        A_5[self.m_5] = self.e_5
        A_6[self.m_6] = self.e_6
        A_7[self.m_7] = self.e_7
        A_8[self.m_8] = self.e_8
        A_9[self.m_9] = self.e_9


        A_0 = F.softmax(A_0, dim=1)
        A_1 = F.softmax(A_1, dim=1)
        A_2 = F.softmax(A_2, dim=1)
        A_3 = F.softmax(A_3, dim=1)
        A_4 = F.softmax(A_4, dim=1)
        A_5 = F.softmax(A_5, dim=1)
        A_6 = F.softmax(A_6, dim=1)
        A_7 = F.softmax(A_7, dim=1)
        A_8 = F.softmax(A_8, dim=1)
        A_9 = F.softmax(A_9, dim=1)
     

        output_0 = torch.matmul(A_0, h0) 
        output_1 = torch.matmul(A_1, h1) 
        output_2 = torch.matmul(A_2, h2) 
        output_3 = torch.matmul(A_3, h3)
        output_4 = torch.matmul(A_4, h4) 
        output_5 = torch.matmul(A_5, h5) 
        output_6 = torch.matmul(A_6, h6) 
        output_7 = torch.matmul(A_7, h7)
        output_8 = torch.matmul(A_8, h8) 
        output_9 = torch.matmul(A_9, h9) 


        if self.out_features is not 3:  

             output = output_0 + output_1 + output_2 + output_3 + output_4 + output_5 + output_7 + output_8 + output_9
             
             # output = torch.cat([output_0, output_1, output_2, output_3], dim = 2)

        else: 

            output = output_0 + output_1 + output_2 + output_3 + output_4 + output_5 + output_7 + output_8 + output_9

            return output + self.bias_2.view(1,1,-1)

        if self.bias is not None:

            return output + self.bias.view(1,1,-1)

        else:

            return output

    def __repr__(self):

        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'









# two order



## Our PoseFormer model was revised from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

import math
import logging
from functools import partial
from collections import OrderedDict
from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model


class HGraphConv(nn.Module):
    """
    High-order graph convolution layer 
    """

    def __init__(self, in_features, out_features, adj, bias=True):
        super(HGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(2, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
  
        self.adj_0 = torch.eye(adj.size(0), dtype=torch.float) # declare self-connections
        self.m_0 = (self.adj_0 > 0)
        self.e_0 =  nn.Parameter(torch.zeros(1, len(self.m_0.nonzero()), dtype=torch.float))
        nn.init.constant_(self.e_0.data, 1)

        self.adj_1 = adj # one_hop neighbors
        self.m_1 = (self.adj_1 > 0)
        self.e_1 = nn.Parameter(torch.zeros(1, len(self.m_1.nonzero()), dtype=torch.float))
        nn.init.constant_(self.e_1.data, 1)
        


        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float)) #self.bias = nn.Parameter(torch.zeros(out_features*4, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
            
            self.bias_2 = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias_2.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        h0 = torch.matmul(input, self.W[0])
        h1 = torch.matmul(input, self.W[1])

    
        A_0 = -9e15 * torch.ones_like(self.adj_0).to(input.device)
        A_1 = -9e15 * torch.ones_like(self.adj_1).to(input.device) # without self-connection 

        A_0[self.m_0] = self.e_0
        A_1[self.m_1] = self.e_1


        A_0 = F.softmax(A_0, dim=1)
        A_1 = F.softmax(A_1, dim=1)

        output_0 = torch.matmul(A_0, h0) 
        output_1 = torch.matmul(A_1, h1) 


        if self.out_features is not 3:  
             output = output_0 + output_1 # output = torch.cat([output_0, output_1, output_2, output_3], dim = 2)
        else: 
            output = output_0 + output_1 
        if self.bias is not None:
            return output + self.bias.view(1,1,-1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


# 4 order

class HGraphConv(nn.Module):
    """
    High-order graph convolution layer 
    """

    def __init__(self, in_features, out_features, adj, bias=True):
        super(HGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(4, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
  
        self.adj_0 = torch.eye(adj.size(0), dtype=torch.float) # declare self-connections
        self.m_0 = (self.adj_0 > 0)
        self.e_0 =  nn.Parameter(torch.zeros(1, len(self.m_0.nonzero()), dtype=torch.float))
        nn.init.constant_(self.e_0.data, 1)

        self.adj_1 = adj # one_hop neighbors
        self.m_1 = (self.adj_1 > 0)
        self.e_1 = nn.Parameter(torch.zeros(1, len(self.m_1.nonzero()), dtype=torch.float))
        nn.init.constant_(self.e_1.data, 1)
        
        self.adj_2 = torch.matmul(self.adj_1, adj) # two_hop neighbors
        self.m_2 = (self.adj_2 > 0)
        self.e_2 = nn.Parameter(torch.zeros(1, len(self.m_2.nonzero()), dtype=torch.float))
        nn.init.constant_(self.e_2.data, 1)

        self.adj_3 = torch.matmul(self.adj_2, adj) # three_hop neighbors
        self.m_3 = (self.adj_3 > 0)
        self.e_3 = nn.Parameter(torch.zeros(1, len(self.m_3.nonzero()), dtype=torch.float))
        nn.init.constant_(self.e_3.data,1)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float)) #self.bias = nn.Parameter(torch.zeros(out_features*4, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
            
            self.bias_2 = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias_2.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        


        h0 = torch.matmul(input, self.W[0])
        h1 = torch.matmul(input, self.W[1])
        h2 = torch.matmul(input, self.W[2])
        h3 = torch.matmul(input, self.W[3])
    
        A_0 = -9e15 * torch.ones_like(self.adj_0).to(input.device)
        A_1 = -9e15 * torch.ones_like(self.adj_1).to(input.device) # without self-connection 
        A_2 = -9e15 * torch.ones_like(self.adj_2).to(input.device)
        A_3 = -9e15 * torch.ones_like(self.adj_3).to(input.device)
         
        A_0[self.m_0] = self.e_0
        A_1[self.m_1] = self.e_1
        A_2[self.m_2] = self.e_2
        A_3[self.m_3] = self.e_3

        A_0 = F.softmax(A_0, dim=1)
        A_1 = F.softmax(A_1, dim=1)
        A_2 = F.softmax(A_2, dim=1)
        A_3 = F.softmax(A_3, dim=1)

        output_0 = torch.matmul(A_0, h0) 
        output_1 = torch.matmul(A_1, h1) 
        output_2 = torch.matmul(A_2, h2) 
        output_3 = torch.matmul(A_3, h3)

        if self.out_features is not 3:  
             output = output_0 + output_1 + output_2 + output_3# output = torch.cat([output_0, output_1, output_2, output_3], dim = 2)
        else: 
            output = output_0 + output_1 + output_2 + output_3
            return output + self.bias_2.view(1,1,-1)
        if self.bias is not None:
            return output + self.bias.view(1,1,-1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


# subgraph



class HGraphConv(nn.Module):
    """
    High-order graph convolution layer 
    """

    def __init__(self, in_features, out_features, adj, bias=True):
        super(HGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(4, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
  
        self.adj_0 = torch.eye(adj.size(0), dtype=torch.float) # declare self-connections
        self.m_0 = (self.adj_0 > 0)
        self.e_0 =  nn.Parameter(torch.zeros(1, len(self.m_0.nonzero()), dtype=torch.float))
        nn.init.constant_(self.e_0.data, 1)

        self.adj_1 = adj # one_hop neighbors
        self.m_1 = (self.adj_1 > 0)
        self.e_1 = nn.Parameter(torch.zeros(1, len(self.m_1.nonzero()), dtype=torch.float))
        nn.init.constant_(self.e_1.data, 1)
        
        self.adj_2 = torch.matmul(self.adj_1, adj) # two_hop neighbors
        self.m_2 = (self.adj_2 > 0)
        self.e_2 = nn.Parameter(torch.zeros(1, len(self.m_2.nonzero()), dtype=torch.float))
        nn.init.constant_(self.e_2.data, 1)

        self.adj_3 = torch.matmul(self.adj_2, adj) # three_hop neighbors
        self.m_3 = (self.adj_3 > 0)
        self.e_3 = nn.Parameter(torch.zeros(1, len(self.m_3.nonzero()), dtype=torch.float))
        nn.init.constant_(self.e_3.data,1)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float)) #self.bias = nn.Parameter(torch.zeros(out_features*4, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
            
            self.bias_2 = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias_2.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        


        h0 = torch.matmul(input, self.W[0])
        h1 = torch.matmul(input, self.W[1])
        h2 = torch.matmul(input, self.W[2])
        h3 = torch.matmul(input, self.W[3])
    
        A_0 = -9e15 * torch.ones_like(self.adj_0).to(input.device)
        A_1 = -9e15 * torch.ones_like(self.adj_1).to(input.device) # without self-connection 
        A_2 = -9e15 * torch.ones_like(self.adj_2).to(input.device)
        A_3 = -9e15 * torch.ones_like(self.adj_3).to(input.device)
         
        A_0[self.m_0] = self.e_0
        A_1[self.m_1] = self.e_1
        A_2[self.m_2] = self.e_2
        A_3[self.m_3] = self.e_3

        A_0 = F.softmax(A_0, dim=1)
        A_1 = F.softmax(A_1, dim=1)
        A_2 = F.softmax(A_2, dim=1)
        A_3 = F.softmax(A_3, dim=1)

        output_0 = torch.matmul(A_0, h0) 
        output_1 = torch.matmul(A_1, h1) 
        output_2 = torch.matmul(A_2, h2) 
        output_3 = torch.matmul(A_3, h3)

        if self.out_features is not 3:  
             output = output_0 + output_1 + output_2 + output_3# output = torch.cat([output_0, output_1, output_2, output_3], dim = 2)
        else: 
            output = output_0 + output_1 + output_2 + output_3
            return output + self.bias_2.view(1,1,-1)
        if self.bias is not None:
            return output + self.bias.view(1,1,-1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
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

        self.adj=adj


        # Linear layer to project the order-specific features to attention scores
        self.attention_projection = nn.Linear(out_features, 1)
        #self.softmax = nn.Softmax(dim=1)


        # Linear layers for query, key, and value projections
        self.query_projection = nn.Linear(out_features, out_features)
        self.key_projection = nn.Linear(out_features, out_features)
        self.value_projection = nn.Linear(out_features, out_features)


        self.attention_projection = nn.Linear(out_features, 1)

        # Softmax for attention weights
        self.softmax = nn.Softmax(dim=1)
        

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
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
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



        gcn_outs=[output_0, output_1, output_2,output_3]


        batch_size, num_joints, features_dim = gcn_outs[0].size()

        # Stack order features along a new dimension
        joint_features_stacked = torch.stack(gcn_outs, dim=2)

        #print("joint_features_stacked",joint_features_stacked.size())
        

        # Project joint-specific features to query and key
        queries = self.query_projection(joint_features_stacked)
        #print("queries",queries.size())

        keys = self.key_projection(joint_features_stacked)
        #print("keys",keys.size())

        # Compute attention scores
        attention_scores = self.attention_projection(torch.tanh(queries + keys))
        #print("attention_scores",attention_scores.size())


        # Reshape back to (batch_size, num_joints, num_orders)
        attention_scores = attention_scores.view(batch_size, num_joints, 4)
        #print("attention_scores view",attention_scores.size())

        # Apply softmax to get attention weights across orders
        joint_attention_weights = self.softmax(attention_scores)
        #print("joint_attention_weights",joint_attention_weights.size())

        # Weighted sum of joint-specific features based on attention weights across orders
        attended_features = torch.sum(joint_features_stacked * joint_attention_weights.unsqueeze(-1), dim=2)
        #print("attended_features",attended_features.size())


        if self.out_features is not 3:  
            output = attended_features    #output = torch.cat([output_0, output_1, output_2, output_3], dim = 2)
        else: 
            output = attended_features

            return output + self.bias_2.view(1,1,-1)
            
        if self.bias is not None:
            return output + self.bias.view(1,1,-1)
        else:
            return output
        

       

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'




class _GraphConv(nn.Module):

    def __init__(self, adj, input_dim, output_dim, p_dropout=None):
        super(_GraphConv, self).__init__()

        self.gconv = HGraphConv(input_dim, output_dim, adj)

        self.relu = nn.ReLU()
        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None
        
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        x = self.gconv(x).transpose(1, 2)
        x = x.transpose(1, 2)
        x = self.norm(x)
        
        if self.dropout is not None:
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        return x






class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AttentionC(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Positional scaling parameters
        self.center_position = nn.Parameter(torch.Tensor([0.5]), requires_grad=False)
        self.scaling_factor = nn.Parameter(torch.Tensor([2.0]), requires_grad=True)


        self.attention_projection=nn.Sequential(

            nn.Linear(32, 1)
        )
        self.softmax = nn.Softmax(dim=1)

        self.norm = nn.LayerNorm(32)
        self.norm2 = nn.LayerNorm(17*32)

    def forward(self, x):

       

        x_reshaped = x.view(-1, x.size(1), 17, 32)


        x_reshaped=self.norm(x_reshaped)

        # Project joint-wise features to attention scores
        attention_scores = self.attention_projection(x_reshaped)

        # Reshape back to (batch_size, num_frames, num_joints)
        attention_scores = attention_scores.view(-1, x.size(1), 17)

        # Apply softmax to get attention weights across joints
        attention_weights = self.softmax(attention_scores)

        # Reshape x to (batch_size, num_frames, num_joints, features_dim) for element-wise multiplication
        x_reshaped = x_reshaped.view(-1, x.size(1), 17, 32)


        
        # Weighted sum of joint-wise features based on attention weights
        #attended_features = torch.sum(x_reshaped * attention_weights.unsqueeze(-1), dim=2)

        attended_features = x_reshaped * attention_weights.unsqueeze(-1)

        
        # Reshape back to (batch_size, num_frames, num_joints * features_dim)
        x = attended_features.view(-1, x.size(1), 17 * 32)

        x=self.norm2(x)

        B, N, C = x.shape

        # Calculate position-dependent scaling factor
        positions = torch.linspace(0, 1, N, device=x.device).unsqueeze(0).unsqueeze(0)
        position_scaling = torch.exp(-self.scaling_factor * (positions - self.center_position) ** 2)

        

        # Linear transformation
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores with positional scaling
        attn = (q @ k.transpose(-2, -1)) * (self.scale * position_scaling)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # Linear transformation and dropout
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attention_projection=nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.softmax = nn.Softmax(dim=1)



    def forward(self, x):

        x_reshaped = x.view(-1, x.size(1), 17, 32)

        # Project joint-wise features to attention scores
        attention_scores = self.attention_projection(x_reshaped)

        # Reshape back to (batch_size, num_frames, num_joints)
        attention_scores = attention_scores.view(-1, x.size(1), p)

        # Apply softmax to get attention weights across joints
        attention_weights = self.softmax(attention_scores)

        # Reshape x to (batch_size, num_frames, num_joints, features_dim) for element-wise multiplication
        x_reshaped = x_reshaped.view(-1, x.size(1), p, 32)


        
        # Weighted sum of joint-wise features based on attention weights
        #attended_features = torch.sum(x_reshaped * attention_weights.unsqueeze(-1), dim=2)

        attended_features = x_reshaped * attention_weights.unsqueeze(-1)

        
        # Reshape back to (batch_size, num_frames, num_joints * features_dim)
        x = attended_features.view(-1, x.size(1), 17 * 32)
        

        

      
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = AttentionC(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x






class GOA_TF(nn.Module):
    def __init__(self, adj,num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=None):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio * num_joints   #### temporal embed_dim is num_joints * spatial embedding dim ratio
        out_dim = num_joints * 3     #### output dimension is num_joints * 3

     
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 6)]  # stochastic depth decay rule


        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(6)])


        self.Temporal_norm = norm_layer(embed_dim)

        ####### A easy way to implement weighted mean
        self.conv1D = torch.nn.Conv1d(in_channels=81, out_channels=1, kernel_size=1)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim , out_dim),
        )

        #### gcn layers 

        hid_dim=32
        coords_dim=(2, 3)
        p_dropout=None


        _gconv_input = [_GraphConv(adj, coords_dim[0], hid_dim, p_dropout=p_dropout)]
        self.gconv_input = nn.Sequential(*_gconv_input)





    def forward(self, x):
        
        x=x[:,::4,:,:]
        x = x.permute(0, 3, 1, 2)     #[256, 2, 9, 17]
        b, c, f, p = x.shape  ##### b is batch size, f is number of frames, p is number of joints
        #[256,9,17,3]

        x = rearrange(x, 'b c f p  -> (b f) p  c', )  #[2304, 17, 2]
        x = self.gconv_input(x)  #[2304, 17, 32]
        x = rearrange(x, '(b f) p c -> b f (p c)', f=f)  
        #x += self.Temporal_pos_embed    #[256, 9, 17*32]
        #x = self.pos_drop(x)
        for blk in self.blocks:  
            x = blk(x) + x              # [256, 9, 544]
        x = self.Temporal_norm(x)
        x = self.conv1D(x)   #[256, 1, 544]
        x = x.view(b, 1, -1)     #[256, 1, 544]
        x = self.head(x) # [256, 1, 51]
        x = x.view(b, 1, p, -1)  #[256, 1, 17, 3]
        return x
      
    

 
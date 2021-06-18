import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedInstanceNorm2d(nn.Module):
    """Masked InstanceNorm2d"""
    def __init__(self, num_features: int, eps: float=1e-05, momentum: float=0.1, affine: bool=False, track_running_stats: bool=False) -> None:
        super(MaskedInstanceNorm2d, self).__init__()
        self.instancenorm2d = nn.InstanceNorm2d(num_features, eps, momentum, affine, track_running_stats)
    
    def forward(self, x, mask):
        """
        Args :
            x : input feature maps (batch_m, c, n, n) (n == n == max_num_stops)
            mask: input feature masks (batch_m, n, n)
        Returns :
            out : instancenorm2d(x) (batch_m, c, n, n)
        """

        # Calculate mean per channel
        x = x.masked_fill(mask == 0, 0)                     # (batch_m, c, max_num_stops, max_num_stops)
        sum = torch.sum(x, dim=(2,3))                       # (batch_m, c)
        count = torch.sum(mask.type(sum.type()), dim=(1,2)) # (batch_m, )
        count = count.unsqueeze(1).repeat(1, x.shape[1])    # (batch_m, c)
        mean = sum / count                                  # (batch_m, c)

        assert count.shape == sum.shape == mean.shape == (x.shape[0], x.shape[1])

        # Pad with mean 
        mean = mean.unsqueeze(2).unsqueeze(3)               # (batch_m, c, 1, 1)
        mean = mean.repeat(1, 1, x.shape[2], x.shape[3])    # (batch_m, c, max_num_stops, max_num_stops)
        mean = mean.masked_fill(mask != 0, 0)               # (batch_m, c, max_num_stops, max_num_stops)
        x = x + mean

        assert mean.shape == x.shape

        return self.instancenorm2d(x)


class Convolution(nn.Module):
    """ Double convolution layer"""
    def __init__(self, in_dim, num_groups):
        super(Convolution, self).__init__()
        self.channel_in = in_dim
        self.num_groups = in_dim // 16 if num_groups is None else num_groups # https://arxiv.org/abs/1803.08494
        self.gamma = 1 # self.gamma = nn.Parameter(torch.zeros(1))   
        self.conv = nn.Sequential(
            nn.GroupNorm(num_groups=self.num_groups, num_channels=in_dim),
            nn.SiLU(),
            nn.Conv2d(in_dim, in_dim*2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=self.num_groups//2, num_channels=in_dim*2),
            nn.SiLU(),
            nn.Conv2d(in_dim*2, in_dim, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        out = self.conv(x)
        return self.gamma*out + x


class PositionEncoding(nn.Module):
    """ Position encoding layer"""
    def __init__(self, router_embbed_dim):
        super(PositionEncoding, self).__init__()
        self.router_embbed_dim = router_embbed_dim
        self.embedd_dim = router_embbed_dim
    
    def forward(self, x):

        """
        Args :
            x : input feature maps (batch_m, router_embbed_dim, n, n) (n == max_num_stops)
        """
        
        _batch_m, _C, height, width = x.size()

        pe = self.positionencoding2d(self.embedd_dim, height, width) # (router_embbed_dim, n, n)

        return x + pe.unsqueeze(0)

    @staticmethod
    def positionencoding2d(embedd_dim:int, height:int, width:int) -> torch.Tensor:
        """
        Args :
            embedd_dim (int): embedding dimension (multiple of 4)
            height (int): height of position encoding
            width (int): width of position encoding
        Return :
            pe (torch.Tensor): position embedding tensor (embedd_dim, height, width)
        """
        
        assert embedd_dim % 4 == 0, f"PE (2*sin/cos) embedding dimension has to be multiple of four (got {embedd_dim})"

        # Pre-allocation
        pe = torch.zeros(embedd_dim, height, width)

        # Each dimension use half of d_model
        # https://github.com/wzlxjtu/PositionalEncoding2D
        embedd_dim = int(embedd_dim / 2)
        
        div_term = torch.exp(torch.arange(0., embedd_dim, 2) * -(math.log(10000.0) / embedd_dim))
        
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        
        pe[0:embedd_dim:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:embedd_dim:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[embedd_dim::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[embedd_dim + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

        return pe


class InputFusion(nn.Module):
    """ Input fusion layer"""
    def __init__(self, in_dim:int, in_dim_0:int, router_embbed_dim:int, aux_embbed_dim:int=16):
        """
        Args :
            in_dim (int): input feature maps dimension (num_1d_features + num_2d_features)
            in_dim_0 (int): auxilary input dimension (num_0d_features)
            router_embbed_dim (int): total number of output channels 
            aux_embbed_dim (int): number of output channels of adaptive kernel
        """
        super(InputFusion, self).__init__()
        self.channel_in = in_dim
        self.channel_in_0 = in_dim_0
        self.channel_out_0 = aux_embbed_dim
        self.pe = PositionEncoding(router_embbed_dim)
        self.norm = nn.InstanceNorm2d(router_embbed_dim)
        self.input = nn.Conv2d(in_dim, router_embbed_dim - aux_embbed_dim, kernel_size=1)
        self.input_0 = nn.Sequential(
            nn.Linear(in_dim_0, in_dim*aux_embbed_dim, bias=False),
            nn.SiLU(),
            nn.LayerNorm(in_dim*aux_embbed_dim),
            nn.Linear(in_dim*aux_embbed_dim, in_dim*aux_embbed_dim)
        )

    def forward(self, x, x_0):
        """
        Args :
            x : input feature maps (batch_m, in_dim, n, n) (n == max_num_stops)
            x_0: auxilary route-level input (batch_m, num_0d_features)
        """

        batch_m, C, height, width = x.size() # height==width==n
        
        # Adaptive kernel using route-level input (auxilary information)
        # https://discuss.pytorch.org/t/setting-custom-kernel-for-cnn-in-pytorch/27176/2
        kernels = self.input_0(x_0) # (m_batch, in_dim * aux_embbed_dim_C) 
        kernels = kernels.reshape(-1, self.channel_in, 1, 1) # (in_dim * aux_embbed_dim_C, in_dim, 1, 1) =:= (out_C, in_C//group, kH, kW)

        # (1) 2d-feature maps from input directly (out_C = router_embbed_dim - aux_embbed_dim)
        out = self.input(x)  # (batch_m, in_dim, n, n) -> (batch_m, router_embbed_dim - aux_embbed_dim, n, n)

        # (2) 2d-feature maps from aux-input using adaptive kernels (out_C = aux_embbed_dim)
        x = x.reshape(1, -1, height, width) # (1, batch_m * in_dim, n, n) =:= (batch, in_C, iH, iW)
        out_0 = F.conv2d(x, kernels, groups=batch_m)
        out_0 = out_0.reshape(batch_m, self.channel_out_0, height, width)
        
        return self.pe(self.norm(torch.cat((out, out_0), 1)))


class Mixing(nn.Module):
    """Fourier-mixing layer"""
    def __init__(self, in_dim, mixing_dim=-1):
        super(Mixing, self).__init__()
        self.in_dim = in_dim
        self.gamma = 1  # self.gamma = nn.Parameter(torch.zeros(1))
        self.norm = nn.InstanceNorm2d(in_dim, affine=True)
        
        if mixing_dim in [-1, 3, -2, 2]:
            self.mixing_dim = mixing_dim
        else:
            raise ValueError("mixing_dim must be (2 or 3), assumed 4D input tensor (batch_m, C, h, w)")

    def forward(self, x, mask):
        # Normalization across channels
        x = self.norm(x)
        
        # Zero padding using mask
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(1) == 0, 0)

        return torch.fft.fft(x, dim=self.mixing_dim).real


class Mixer(nn.Module):
    """ Mixer block"""
    def __init__(self, in_dim, summary_dim=-1, dropout=0.):
        super(Mixer, self).__init__()
        self.channel_in = in_dim
        self.mixing = Mixing(in_dim, summary_dim)
        self.convolution = Convolution(in_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        out = self.mixing(x, mask)
        out = self.convolution(out)
        out = self.dropout(out)
        return out


class SelfAttention(nn.Module):
    """ Self-attention layer"""
    def __init__(self, in_dim, summary_dim=-1):
        super(SelfAttention, self).__init__()
        self.channel_in = in_dim
        self.summary_dim = summary_dim
        self.gamma = 1  # self.gamma = nn.Parameter(torch.zeros(1))
        self.norm = nn.InstanceNorm2d(in_dim, affine=True) 
        self.queries = nn.Conv2d(in_dim, in_dim//8, kernel_size=1, bias=False)
        self.keys = nn.Conv2d(in_dim ,in_dim//8, kernel_size=1, bias=False)
        self.values = nn.Conv2d(in_dim , in_dim, kernel_size= 1, bias=False)

        if summary_dim == -1 or summary_dim == 3: 
            self.pool = nn.AdaptiveAvgPool2d((None, 1))  # (batch_m, C, n, n) -> (batch_m, C, n, 1)
        elif summary_dim == -2 or summary_dim == 2:
            self.pool = nn.AdaptiveAvgPool2d((1, None))  # (batch_m, C, n, n) -> (batch_m, C, 1, n)
        else:
            raise ValueError("summary_dim must be (2 or 3), assumed 4D input tensor (batch_m, C, h, w)")

    def forward(self, x, mask):
        """
        Args :
            x : input feature maps (batch_m, c, n, n) (n == max_num_stops)
            mask: input feature masks (batch_m, n, n)
        Returns :
            out : self attention value + input feature (batch_m, c, n, n)
            energy: energy values (batch_m, n, n) 
        """

        m_batch, C, height, width = x.size() # height==width==n

        # Normalization across channels
        x = self.norm(x)
    
        # Queries, keys, and values
        queries = self.queries(self.pool(x)).reshape(m_batch,-1,height) # (batch_m, C//8, n, 1) or (batch_m, C//8, 1, n) -> (batch_m, C//8, n)
        keys =  self.keys(self.pool(x)).reshape(m_batch,-1,height)      # (batch_m, C//8, n, 1) or (batch_m, C//8, 1, n) -> (batch_m, C//8, n)
        values = self.values(x)                                         # (batch_m, C, n, n) 

        # Attention softmax(Q^T*K)
        energy =  torch.bmm(queries.permute(0,2,1), keys) # (batch_m, n, n)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        attention = torch.softmax(energy, dim=-1)         # (batch_m, n, n)

        # Output
        if self.summary_dim == -1 or self.summary_dim == 3:
            out = torch.einsum("mik,mckj->mcij", attention, values)  # (batch_m, C, n, n)
        else:
            out = torch.einsum("mjk,mcik->mcij", attention, values)  # (batch_m, C, n, n)

        out = self.gamma*out + x
        
        return out, energy


class CrossAttention(nn.Module):
    """Cross attention layer"""
    def __init__(self, in_dim:int, contraction_factor:int=2):
        super(CrossAttention, self).__init__()
        self.channel_in = in_dim
        self.alpha = contraction_factor
        self.gamma = 1  # self.gamma = nn.Parameter(torch.zeros(1))
        self.norm = nn.InstanceNorm2d(in_dim, affine=True) 
        # self.queries = nn.Conv2d(in_dim, in_dim//8, kernel_size=1, bias=False)
        # self.keys = nn.Conv2d(in_dim ,in_dim//8, kernel_size=1, bias=False)
        # self.values = nn.Conv2d(in_dim , in_dim, kernel_size= 1, bias=False)
        self.qkv = nn.Conv2d(in_dim, 2 * (in_dim//self.alpha) + in_dim, kernel_size=1, bias=False)


    def forward(self, x, mask):
        """
        Args :
            x : input feature maps (batch_m, c, n, n) (n == max_num_stops)
            mask: input feature masks (batch_m, n, n)
        Returns :
            out : self attention value + input feature (batch_m, c, n, n)
            energy: energy values (batch_m, h, w, h+w) height==width==n
        """

        m_batch, C, height, width = x.size() # height==width==n

        # Normalization across channels
        out = self.norm(x) # (batch_m, C, h, w)
        out = self.qkv(out)  # (batch_m, C//8 + C//8 + C, h, w)
        queries, keys, values = torch.split(out, [self.channel_in//self.alpha,  self.channel_in//self.alpha,  self.channel_in], dim=1) # (batch_m, C//8, h, w), (batch_m, C//8, h, w), (batch_m, C, h, w)

        # Queries
        # queries = self.queries(x) # (batch_m, C//8, h, w)
        queries_H = queries.permute(0,3,1,2).contiguous().view(m_batch*width, -1, height).permute(0, 2, 1) # (batch_m, C//8, h, w) -> (batch_m * w, C//8, h) -> (batch_m * w, h, C//8)
        queries_W = queries.permute(0,2,1,3).contiguous().view(m_batch*height, -1, width).permute(0, 2, 1) # (batch_m, C//8, h, w) -> (batch_m * h, C//8, w) -> (batch_m * h, w, C//8)

        # Keys
        # keys = self.keys(x) # (batch_m, C//8, h, w)
        keys_H = keys.permute(0,3,1,2).contiguous().view(m_batch*width, -1, height) # (batch_m, C//8, h, w) -> (batch_m * w, C//8, h)
        keys_W = keys.permute(0,2,1,3).contiguous().view(m_batch*height,- 1, width) # (batch_m, C//8, h, w) -> (batch_m * h, C//8, w)

        # Values
        # values = self.values(x) # (batch_m, C, h, w)
        values_H = values.permute(0,3,1,2).contiguous().view(m_batch*width, -1, height) # (batch_m, C, h, w) -> (batch_m * w, C, h)
        values_W = values.permute(0,2,1,3).contiguous().view(m_batch*height, -1, width) # (batch_m, C, h, w) -> (batch_m * h, C, w)

        # Energy
        energy_H = (torch.bmm(queries_H, keys_H) + self.INF(m_batch, height, width)).view(m_batch, width, height, height).permute(0,2,1,3) # (batch_m * w, h, h) -> (batch_m, h, w, h)
        energy_W = torch.bmm(queries_W, keys_W).view(m_batch, height, width, width) # (batch_m * h, w, w) -> (batch_m, h, w, w)
        energy = torch.cat([energy_H, energy_W], 3) # (batch_m, h, w, h+w)

        if mask is not None:
            energy = energy.masked_fill(mask.unsqueeze(-1) == 0, float("-1e20")) # (batch_m, h, w, h+w)

        # Attention
        att = torch.softmax(energy, dim=-1)         # (batch_m, h, w, h+w)
        att_H = att[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batch*width, height, height) # (batch_m, h, w, h) -> (batch_m, w, h, h) -> (batch_m * w, h, h)
        att_W = att[:,:,:,height:height+width].contiguous().view(m_batch*height, width, width) # (batch_m, h, w, w) -> (batch_m * h, w, w)

        # Out
        out_H = torch.bmm(values_H, att_H.permute(0, 2, 1)).view(m_batch,width,-1,height).permute(0,2,3,1) # (batch_m * w, C, h) @ (batch_m * w, h, h) -> (batch_m, w, C, h) -> (batch_m, C, h, w)
        out_W = torch.bmm(values_W, att_W.permute(0, 2, 1)).view(m_batch,height,-1,width).permute(0,2,1,3) # (batch_m * h, C, w) @ (batch_m * h, w, w) -> (batch_m, h, C, w) -> (batch_m, C, h, w)
        out = out_H + out_W # (batch_m, C, h, w)
        return out + x, energy
    
    @staticmethod
    def INF(B:int,H:int,W:int):
        return -torch.diag(torch.tensor(float("inf")).repeat(H),0).unsqueeze(0).repeat(B*W,1,1)


class CrossMultiHeadAttention(nn.Module):
    """Cross multi-head attention layer"""
    def __init__(self, in_dim:int, heads:int=2, num_groups:int=2, contraction_factor:int=2):
        super(CrossMultiHeadAttention, self).__init__()
        self.channel_in = in_dim
        self.heads = heads
        self.num_groups = num_groups
        self.alpha = contraction_factor
        self.gamma = 1  # self.gamma = nn.Parameter(torch.zeros(1))
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=in_dim) 
        self.qkv = nn.Conv2d(in_dim, 2 * (in_dim//self.alpha) + in_dim, kernel_size=1, bias=False)

        assert heads < in_dim, "number of attention-heads has to be less than the router dimension"
        assert heads < in_dim//contraction_factor, "number of attention-heads has to be less than the router_dimension // contraction_factor "

    def forward(self, x, mask):
        """
        Args :
            x : input feature maps (batch_m, c, n, n) (n == max_num_stops)
            mask: input feature masks (batch_m, n, n)
        Returns :
            out : self attention value + input feature (batch_m, c, n, n)
            energy: energy values (batch_m, h, w, h+w) height==width==n
        """

        m_batch, C, height, width = x.size() # height==width==n

        # Normalization across channels
        out = self.norm(x) # (batch_m, C, h, w)
        out = self.qkv(out)  # (batch_m, C//a + C//a + C, h, w)
        queries, keys, values = torch.split(out, [self.channel_in//self.alpha,  self.channel_in//self.alpha,  self.channel_in], dim=1) # (batch_m, C//a, h, w), (batch_m, C//a, h, w), (batch_m, C, h, w)

        # Split into-heads
        queries = queries.view(m_batch, self.heads, -1, height, width) # (batch_m, C//a, h, w) -> (batch_m, heads, C_head//a, h, w)
        keys = keys.view(m_batch, self.heads, -1, height, width)       # (batch_m, C//a, h, w) -> (batch_m, heads, C_head//a, h, w)
        values = values.view(m_batch, self.heads, -1, height, width)   # (batch_m, C//a, h, w) -> (batch_m, heads, C_head, h, w)

        # Queries
        queries_H = (
            queries.permute(0,1,4,2,3).contiguous()                 # (batch_m, heads, C_head//a, h, w) -> (batch_m, heads, w, C_head//a, h)
                   .view(m_batch*self.heads*width, -1, height)      # (batch_m, heads, w, C_head//a, h) -> (batch_m * heads * w, C_head//a, h)
                   .permute(0, 2, 1)                                # (batch_m * heads * w, C_head//a, h) -> (batch_m * heads * width, h, C_head//a)
        )
        queries_W = (
            queries.permute(0,1,3,2,4).contiguous()                 # (batch_m, heads, C_head//a, h, w) -> (batch_m, heads, h, C_head//a, w)
                   .view(m_batch*self.heads*height, -1, width)      # (batch_m, heads, h, C_head//a, w) -> (batch_m * heads * h, C_head//a, w)
                   .permute(0, 2, 1)                                # (batch_m * heads * h, C_head//a, w) -> (batch_m * heads * h, w, C_head//a)
        )

        # Keys
        keys_H = (
            keys.permute(0,1,4,2,3).contiguous()                    # (batch_m, heads, C_head//a, h, w) -> (batch_m, heads, w, C_head//a, h)
                .view(m_batch*self.heads*width, -1, height)         # (batch_m, heads, w, C_head//a, h) -> (batch_m * heads * w, C_head//a, h)
        )
        keys_W = (
            keys.permute(0,1,3,2,4).contiguous()                    # (batch_m, heads, C_head//a, h, w) -> (batch_m, heads, h, C_head//a, w)
                .view(m_batch*self.heads*height, -1, width)         # (batch_m, heads, h, C_head//a, w) -> (batch_m * heads * h, C_head//a, w)
        )

        # Values
        values_H = (
            values.permute(0,1,4,2,3).contiguous()                   # (batch_m, heads, C_head, h, w) -> (batch_m, heads, w, C_head, h)
                  .view(m_batch*self.heads*width, -1, height)        # (batch_m, heads, w, C_head, h) -> (batch_m * heads * w, C_head, h)
        )
        values_W = (
            values.permute(0,1,3,2,4).contiguous()                   # (batch_m, heads, C_head, h, w) -> (batch_m, heads, h, C_head, w)
                  .view(m_batch*self.heads*height, -1, width)        # (batch_m, heads, h, C_head, w) -> (batch_m * heads * h, C_head, w)
        )

        # Energy
        energy_H = (
            (torch.bmm(queries_H, keys_H) + self.INF(m_batch*self.heads, height, width)) # (batch_m * heads * width, h, h) 
            .view(m_batch, self.heads, width, height, height)       # (batch_m * heads * width, h, h) -> (batch_m, heads, width, h, h)
            .permute(0,1,3,2,4)                                     # (batch_m, heads, width, h, h) -> (batch_m, heads, h, width, h)
        )
        energy_W = (
             torch.bmm(queries_W, keys_W)                           # (batch_m * heads * h, w, w)
                  .view(m_batch, self.heads, height, width, width)  # (batch_m * heads * h, w, w) -> (batch_m, heads, h, w, w)
        )
        energy = torch.cat([energy_H, energy_W], -1)                # (batch_m, heads, h, w, h+w)

        if mask is not None:
            energy = energy.masked_fill(mask.unsqueeze(-1).unsqueeze(1) == 0, float("-1e20")) # (batch_m, heads, h, w, h+w)

        # Attention
        att = torch.softmax(energy, dim=-1)     # (batch_m, heads, h, w, h+w)
        att_H = (
            att[:,:,:,:,0:height]               # (batch_m, heads, h, w, h)
            .permute(0,1,3,2,4).contiguous()    # (batch_m, heads, w, h, h)
            .view(m_batch*self.heads*width, height, height) # (batch_m * heads * w, h, h)
        )
        att_W = (
            att[:,:,:,:,height:height+width]    # (batch_m, heads, h, w, w)
            .contiguous()
            .view(m_batch*self.heads*height, width, width) # (batch_m * heads * h, w, w)
        )

        # Out
        out_H = (
            torch.bmm(values_H, att_H.permute(0, 2, 1))    # (batch_m * heads * w, C_head, h)
                 .view(m_batch,self.heads,width,-1,height) # (batch_m, heads, w, C_head, h)
                 .permute(0,1,3,4,2).contiguous()          # (batch_m, heads, w, C_head, h) -> (batch_m, heads, C_head, h, w)
                 .view(m_batch, -1, height, width)         # (batch_m, heads, w, C_head, h) -> (batch_m, C, h, w)
        )
        out_W = (
            torch.bmm(values_W, att_W.permute(0, 2, 1))    # (batch_m * heads * h, C_head, w)
                 .view(m_batch,self.heads,height,-1,width) # (batch_m, heads, h, C_head, w)
                 .permute(0,1,3,2,4).contiguous()          # (batch_m, heads, h, C_head, w) -> (batch_m, heads, C_head, h, w)
                 .view(m_batch, -1, height, width)         # (batch_m, heads, C_head, h, w) -> (batch_m, C, h, w)
        )

        out = out_H + out_W # (batch_m, C, h, w)

        return out + x, energy
    
    @staticmethod
    def INF(B:int,H:int,W:int):
        return -torch.diag(torch.tensor(float("inf")).repeat(H),0).unsqueeze(0).repeat(B*W,1,1)


class Router(nn.Module):
    """ Router block"""
    def __init__(self, in_dim, summary_dim=-1, dropout=0.):
        super(Router, self).__init__()
        self.channel_in = in_dim
        self.attention = SelfAttention(in_dim, summary_dim)
        self.convolution = Convolution(in_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        out, _ = self.attention(x, mask)
        out = self.convolution(out)
        out = self.dropout(out)
        return out


class RouterV4(nn.Module):
    """ Router block V4"""
    def __init__(self, in_dim, dropout=0.):
        super(RouterV4, self).__init__()
        self.channel_in = in_dim
        self.attention = CrossAttention(in_dim)
        self.convolution = Convolution(in_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        out, _ = self.attention(x, mask)
        out = self.convolution(out)
        out = self.dropout(out)
        return out


class RouterV5(nn.Module):
    """ Router block V5"""
    def __init__(self, in_dim:int, num_heads:int=2, num_groups:int=2, contraction_factor:int=2, dropout:float=0.):
        super(RouterV5, self).__init__()
        self.channel_in = in_dim
        self.attention = CrossMultiHeadAttention(in_dim, num_heads, num_groups, contraction_factor)
        self.convolution = Convolution(in_dim, num_groups)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        out, _ = self.attention(x, mask)
        out = self.convolution(out)
        out = self.dropout(out)
        return out
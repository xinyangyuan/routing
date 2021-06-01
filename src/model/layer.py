import torch
import torch.nn as nn

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
    def __init__(self, in_dim):
        super(Convolution, self).__init__()
        self.channel_in = in_dim
        self.gamma = 1 # self.gamma = nn.Parameter(torch.zeros(1))   
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(in_dim, affine=True),
            nn.ReLU(),
            nn.Conv2d(in_dim, in_dim//2, kernel_size=1, bias=False),
            nn.InstanceNorm2d(in_dim//2, affine=True),
            nn.ReLU(),
            nn.Conv2d(in_dim//2, in_dim, kernel_size=1, bias=False),
        )

    def forward(self, x):
        out = self.conv(x)
        return self.gamma*out + x

# class Summary(nn.Module):
#     """ Summary layer """
#     def __init__(self, summary_dim=-1, summary_strategy="pool", in_dim=None):
#         super(Summary, self).__init__()
        
#         if summary_strategy == "pool":
#             if summary_dim == -1 or summary_dim == 3:
#                 self.summary = nn.AdaptiveAvgPool2d((None, 1)) # (batch_m, C, n, n) -> (batch_m, C, n, 1) 
#             elif summary_dim == -2 or summary_dim == 2:
#                 self.summary = nn.AdaptiveAvgPool2d((1, None)) # (batch_m, C, n, n) -> (batch_m, C, 1, n) 
#             else
#                 raise ValueError("summary_dim must be (2 or 3), assumed 4D input tensor (batch_m, C, h, w)")

#         elif s
    
#     def forward(self, x):
#         return self.summary(x)

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
    def __init__(self, in_dim, summary_dim=-1, dropout=0):
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

class Router(nn.Module):
    """ Router block"""
    def __init__(self, in_dim, summary_dim=-1, dropout=0):
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

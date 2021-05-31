import torch
import torch.nn as nn
from torch.types import Number

from model.layer import SelfAttention, Router

class RouteNet(nn.Module):
    """ RouteNet"""
    def __init__(self, in_dim=42, router_embbed_dim=128, num_routers=16, dropout=0):
        super(RouteNet, self).__init__()
        self.input_size = in_dim
        self.channel_in = in_dim
        self.router_embbed_dim = router_embbed_dim # C
        self.num_routers = num_routers
        self._summary_dims = [-1, -2] # -1 -> row-wise; -2 -> column-wise
   
        self.inp_layer = nn.Conv2d(in_dim, router_embbed_dim, kernel_size=1)
        self.out_layer = SelfAttention(router_embbed_dim)
        self.routers = nn.ModuleList([
            Router(router_embbed_dim, self._summary_dims[i % len(self._summary_dims)], dropout) for i in range(num_routers)
        ])
                
    def forward(self, x, mask):
        """
        Args :
            x : input feature maps (batch_m, n, n, input_size) (n == max_num_stops)
            mask: input feature masks (batch_m, n, n)
        Returns :
            out : log_softmax prediction (batch_m, n, n)
        """

        # Input layer
        out = x.permute(0,3,1,2) # (batch_m, input_size, n, n)        
        out = self.inp_layer(out)    # (batch_m, C, n, n)

        # if mask is not None:
        #     out = out.masked_fill(mask.unsqueeze(1) == 0, 0)
        
        # Router layers
        for router in self.routers:
            out = router(out, mask)

        # Output layer
        # note: log_softmax provides numerical stability, which log(0) is kept at
        # value of float("1e-20") instead of output -inf or nan.
        _, out = self.out_layer(out, mask)  # (batch_m, n, n)
        out = nn.functional.log_softmax(out, dim=-1)
        return out


def accuracy(outputs, targets) -> Number:
    """
    Compute the accuracy, given the outputs and targets for all images.
    Args:
        outputs: (torch.Tensor) dimension (batch_size, num_stops, num_stops) - log softmax output of the model
        targets: (torch.Tensor) dimension (batch_size, num_stops) - where each example in batch is list-like [2, 1, 3, 6, 4, 5]
    Returns: (float) accuracy in [0,1]
    """

    preds = outputs.reshape(-1, outputs.shape[2]).argmax(dim=1)
    targets = targets.reshape(-1) 

    return (torch.sum(preds == targets).float() / len(targets)).item()


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}
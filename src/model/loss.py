import torch
import torch.nn.functional as F

# Adapted from LabelSmoothingCrossEntropy loss
# https://amaarora.github.io/2020/07/18/label-smoothing.html
# fastai https://github.com/fastai/fastai2/blob/master/fastai2/layers.py#L338
class LabelSmoothingCrossEntropy(torch.nn.Module):
    """ CrossEntropy Loss with label smoothing.
    """

    def __init__(self, ε:float=0.1, reduction='mean'):
        super().__init__()
        self.ε, self.reduction = ε, reduction
    
    def forward(self, output, target):
        # number of classes
        c = output.size()[-1]
        
        log_preds = F.log_softmax(output, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        
        # (1-ε) * H(q,p) + ε * H(u,p)
        return (1-self.ε)*nll + self.ε*(loss/c) 
    
    @staticmethod
    def reduce_loss(loss, reduction='mean'):
        return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss

class LabelSmoothingNLLLoss(torch.nn.Module):
    """ CrossEntropy Loss with label smoothing.
    """
    
    def __init__(self, ε:float=0.1, reduction:str='mean', ignore_index:int=-100):
        super().__init__()
        self.ε = ε
        self.reduction = reduction
        self.ignore_index =ignore_index
    
    def forward(self, output, target):
        """
        Args :
            output: (torch.Tensor) model prediction reshaped (batch_m * n, n) (n == max_num_stops)
            target: (torch.Tensor) target labels (batch_m * n, 1)
        """

        # Number of stops/classes
        c = output.size()[-1]

        # Calculate H(u,p) and H(q,p)
        loss = self.reduce_loss(-output.sum(dim=-1), self.reduction) # H(u,p)
        nll = F.nll_loss(output, target, reduction=self.reduction, ignore_index=self.ignore_index) # H(q,p)

        # (1-ε) * H(q,p) + ε * H(u,p)
        return (1-self.ε)*nll + self.ε*(loss/c) 
    
    @staticmethod
    def reduce_loss(loss, reduction='mean'):
        return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss
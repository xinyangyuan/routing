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
            target: (torch.Tensor) target labels (batch_m * n)
        """

        # Number of stops/classes
        c = output.size()[-1]

        # Remove index
        loss = -output[target != self.ignore_index] # remove padded starting-stops
        loss[loss == 1e20] = 0 # remove masked stops from F.log_softmax()

        # Calculate H(u,p) and H(q,p)
        loss = self.reduce_loss(loss.sum(dim=-1), self.reduction) # H(u,p)
        nll = F.nll_loss(output, target, reduction=self.reduction, ignore_index=self.ignore_index) # H(q,p)

        # (1-ε) * H(q,p) + ε * H(u,p)
        return (1-self.ε)*nll + self.ε*(loss/c) 
    
    @staticmethod
    def reduce_loss(loss, reduction='mean'):
        return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss

class EditDistanceLoss(torch.nn.Module):
    """ Pseudo-edit distance Loss.
    """
    
    def __init__(self, ignore_index:int=-100):
        super().__init__()
        self.ignore_index =ignore_index
    
    def forward(self, output, target, travel_time):
        """
        Args :
            output: (torch.Tensor) model prediction reshaped (batch_m * n, n) (n == max_num_stops)
            target: (torch.Tensor) target labels (batch_m * n)
            travel_time: (torch.Tensor) travel time between stop-pair (batch_m * n, n)
        """

        # [78, 78, 89, 88, 79, 89]
        # [1,23,33,44,33,223,334,32]
        # travel_time [78][80] 

        # Number of stops/classes
        c = output.size()[-1]

        # Remove index
        loss = -output[target != self.ignore_index] # remove padded starting-stops
        loss[loss == 1e20] = 0 # remove masked stops from F.log_softmax()

        # Calculate H(u,p) and H(q,p)
        loss = self.reduce_loss(loss.sum(dim=-1), self.reduction) # H(u,p)
        nll = F.nll_loss(output, target, reduction=self.reduction, ignore_index=self.ignore_index) # H(q,p)

        # (1-ε) * H(q,p) + ε * H(u,p)
        return (1-self.ε)*nll + self.ε*(loss/c) 


class TimePenalty(torch.nn.Module):
    """ An added time-penalty for prediction. 
        The penalty is calculated using KL-Divergence
    """
    
    def __init__(self, ignore_index:int=-100):
        super().__init__()
        self.ignore_index =ignore_index
    
    def forward(self, output:torch.Tensor, target: torch.Tensor, travel_time:torch.Tensor):
        """
        Args :
            output: (torch.Tensor) model prediction reshaped (batch_m * n, n) (n == max_num_stops)
            target: (torch.Tensor) target labels (batch_m * n)
            travel_time: (torch.Tensor) travel time between stop-pair (batch_m * n, n)
        """

        # squash travel_time to range of [0,1]
        # travel_time = (travel_time - travel_time.min(dim=-1, keepdim=True)) / (travel_time.max(dim=-1, keepdim=True) - travel_time.min(dim=-1, keepdim=True))
        # target = 1 - travel_time # target probability

        # ignore padding 
        travel_time = travel_time[target != self.ignore_index]
        output = output[target != self.ignore_index]

        # penalty
        penalty = F.kl_div(output, F.softmax(-travel_time, dim=-1), reduction="batchmean")

        return penalty 
    
class LabelSmoothingNLLLossWithTimePenalty(torch.nn.Module):
    """ label smoothed CrossEntropy with additional Time Penalty. 
    """

    def __init__(self, alpha:float=0.1, ignore_index:int=-100):
        """
        Args :
            alpha: (float) the weighting paramter for time penalty
            ignore_index: (int) token index for padded target
        """

        super().__init__()
        self.alpha = alpha
        self.ignore_index =ignore_index
        self.label_smooth_nll_loss = LabelSmoothingNLLLoss(ignore_index=ignore_index)
        self.time_penalty = TimePenalty(ignore_index=ignore_index)

    def forward(self, output:torch.Tensor, target: torch.Tensor, travel_time:torch.Tensor):
        """
        Args :
            output: (torch.Tensor) model prediction reshaped (batch_m * n, n) (n == max_num_stops)
            target: (torch.Tensor) target labels (batch_m * n)
            travel_time: (torch.Tensor) travel time between stop-pair (batch_m * n, n)
        """

        return self.label_smooth_nll_loss(output, target) + self.alpha * self.time_penalty(output, target, travel_time)


if __name__ == "__main__":
    
    loss = LabelSmoothingNLLLoss(ignore_index=-100)
    outputs = F.log_softmax(torch.rand(4, 200, 200), dim=-1)
    targets = torch.randint(200, (4, 200))

    outputs = outputs.reshape(-1, outputs.shape[2])
    targets = targets.reshape(-1)

    print(loss(outputs, targets).item())
    
    M = torch.randn(4,4)

    print(M.shape)

    idx_mask = torch.Tensor([1,3,-100,-100])

    print(M)
    print(M[idx_mask != -100])

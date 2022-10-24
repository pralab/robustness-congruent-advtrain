import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

class PCTLoss(CrossEntropyLoss):
    def __init__(self, old_output_clean,
                gamma1=1, alpha1=0, beta1=1):
        super(PCTLoss, self).__init__()
        self.old_output_clean = old_output_clean
        self.gamma1 = gamma1
        self.alpha1 = alpha1
        self.beta1 = beta1
    
    def forward(self, model_output: Tensor, target: Tensor, 
                batch_idx, batch_size, curr_batch_dim) -> Tensor:
        
        old_outs = self.old_output_clean[batch_idx*batch_size:batch_idx*batch_size + curr_batch_dim]
        loss_ce = F.cross_entropy(model_output, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction,
                               label_smoothing=self.label_smoothing)
        
        # apply a weighting for each training sample based on old model outputs
        old_preds_clean = torch.argmax(old_outs, dim=1)
        f_pc = self.alpha1 + self.beta1*((target == old_preds_clean).float()) 
        # distance bw new and old model outputs
        D_pc = F.mse_loss(model_output, old_outs, reduction='none').sum(dim=1)
        # weight differently samples classified correctly by the old model
        loss_pc = f_pc*D_pc

        # # combine CE loss and PCT loss
        loss = loss_ce + self.gamma1*loss_pc

        return loss.mean(), loss_ce.mean(), loss_pc.mean()

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module
import torch.nn.functional as F

class PCTLoss(Module):
    def __init__(self, old_output_clean,
                gamma1=1, alpha1=0, beta1=1):
        super(PCTLoss, self).__init__()
        self.old_output_clean = old_output_clean
        self.gamma1 = gamma1
        self.alpha1 = alpha1
        self.beta1 = beta1

        self.ce = CrossEntropyLoss()
    
    def forward(self, model_output: Tensor, target: Tensor, 
                batch_idx, batch_size, curr_batch_dim) -> Tensor:
        
        old_outs = self.old_output_clean[batch_idx*batch_size:batch_idx*batch_size + curr_batch_dim]
        # loss_ce = F.cross_entropy(model_output, target, reduction='none')
        loss_ce = self.ce(model_output, target.long())


        # apply a weighting for each training sample based on old model outputs
        old_preds = torch.argmax(old_outs, dim=1)
        f_pc = (self.alpha1 + self.beta1 * ((old_preds == target).type(loss_ce.dtype)))
        D_pc = torch.mean((model_output - old_outs).pow(2), dim=1) / 2
        loss_pc = torch.mean(f_pc * D_pc)

        # # combine CE loss and PCT loss
        loss = loss_ce + self.gamma1*loss_pc

        return loss, loss_ce, loss_pc

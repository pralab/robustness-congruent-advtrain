import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module
import torch.nn.functional as F
from typing import Union

class BaseLoss(Module):
    def __init__(self, keep_loss_path=True):
        super(BaseLoss, self).__init__()
        self.keep_loss_path = keep_loss_path
        self.loss_path = {}

    def _add_loss_term(self, key: Union[str, tuple]):
        if isinstance(key, str):
            self.loss_path[key] = []
        else:
            self.loss_path['tot'] = []
            for k in key:
                self.loss_path[k] = []


    def _update_loss_path(self, losses, keys):
        for key, loss in list(zip(keys, losses)):
            self.loss_path[key].append(loss.item())


class BasePCTLoss(BaseLoss):
    def __init__(self, keep_loss_path=True):
        super(BasePCTLoss, self).__init__()
        self.keep_loss_path = keep_loss_path
        self.loss_path = {}

    @staticmethod
    def _logits_to_corrects(logits, target, batch_idx, batch_size, curr_batch_dim):
        outs = logits[batch_idx * batch_size:batch_idx * batch_size + curr_batch_dim]
        preds = torch.argmax(outs, dim=1)
        correct = (preds == target)
        return outs, correct

    def _compute_loss_pc(self, output, old_correct, old_output,
                         alpha=None, beta=None):
        alpha = self.alpha1 if alpha is None else alpha
        beta = self.beta1 if beta is None else beta

        f_pc = self.alpha1 + self.beta1 * old_correct
        D_pc = torch.mean((output - old_output).pow(2), dim=1) / 2
        loss_pc = torch.mean(f_pc * D_pc)
        return loss_pc




class MyCrossEntropyLoss(BaseLoss, CrossEntropyLoss):
    def __init__(self):
        super(MyCrossEntropyLoss, self).__init__()
        key = 'ce'
        self._add_loss_term(key)
        self.loss_keys = tuple(self.loss_path.keys())

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        loss = F.cross_entropy(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction,
                               label_smoothing=self.label_smoothing)


        if self.keep_loss_path:
            self.update_loss_path(loss)

        return loss

    def update_loss_path(self, loss):
        self._update_loss_path((loss,), self.loss_keys)


class PCTLoss(BasePCTLoss):
    def __init__(self, old_output_clean,
                gamma1=1, alpha1=0, beta1=1):
        super(PCTLoss, self).__init__()
        self.old_output_clean = old_output_clean
        self.gamma1 = gamma1
        self.alpha1 = alpha1
        self.beta1 = beta1

        self.ce = CrossEntropyLoss()

        keys = ('ce', 'pc')
        self._add_loss_term(keys)
        self.loss_keys = tuple(self.loss_path.keys())


    
    def forward(self, model_output: Tensor, target: Tensor, 
                batch_idx, batch_size, curr_batch_dim) -> Tensor:
        loss_ce = self.ce(model_output, target.long())

        # apply a weighting for each training sample based on old model outputs
        old_outs, old_correct = self._logits_to_corrects(self.old_output_clean,
                                                         target, batch_idx,
                                                         batch_size, curr_batch_dim)
        loss_pc = self._compute_loss_pc(model_output, old_correct, old_outs)

        # # combine CE loss and PCT loss
        loss = loss_ce + self.gamma1*loss_pc

        if self.keep_loss_path:
            self._update_loss_path((loss, loss_ce, loss_pc), self.loss_keys)

        return loss, loss_ce, loss_pc




class MixedPCTLoss(BasePCTLoss):
    def __init__(self, output1, output2,
                 gamma1=1, alpha1=0, beta1=1, only_nf=False):
        """

        :param output1: outputs of old model on the training set
        :param output2: outputs of the new model from which we finetune
        :param gamma1: to deprecate
        :param alpha1: pure distillation, to deprecate
        :param beta1: bonus for NF, only one useful here
        """
        super(MixedPCTLoss, self).__init__()
        self.output1 = output1
        self.output2 = output2
        self.gamma1 = gamma1
        self.alpha1 = alpha1
        self.beta1 = beta1
        self.only_nf = only_nf

        keys = ('old_mse', 'new_mse')
        self._add_loss_term(keys)
        self.loss_keys = tuple(self.loss_path.keys())


    def forward(self, model_output: Tensor, target: Tensor,
                batch_idx: int, batch_size: int, curr_batch_dim: int) -> Tensor:

        outs1, correct1 = self._logits_to_corrects(self.output1, target,
                                                   batch_idx, batch_size, curr_batch_dim)
        outs2, correct2 = self._logits_to_corrects(self.output2, target,
                                                   batch_idx, batch_size, curr_batch_dim)

        if self.only_nf:
            correct1 = correct1.logical_and(correct2.logical_not())

        # apply a weighting for each training sample based on old model outputs
        f_pc = self.beta1 * correct1
        D_pc = torch.mean((model_output - outs1).pow(2), dim=1) / 2
        loss_pc1 = torch.mean(f_pc * D_pc)

        #f_pc = ((correct2.type(loss_ce.dtype)))
        D_pc = torch.mean((model_output - outs2).pow(2), dim=1) / 2
        loss_pc2 = torch.mean(D_pc)


        # # combine CE loss and PCT loss
        #loss = loss_ce + self.gamma1 * loss_pc

        loss = loss_pc1 + loss_pc2

        if self.keep_loss_path:
            self._update_loss_path((loss, loss_pc1, loss_pc2), self.loss_keys)

        return loss, loss_pc1, loss_pc2


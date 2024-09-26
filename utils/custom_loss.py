import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module
import torch.nn.functional as F
from typing import Union

class BaseLoss(Module):
    """
    This class is just used to save loss information during training
    """
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

    # def _compute_loss_pc(self, output, old_correct, old_output,
    #                      alpha=None, beta=None):
    #     alpha = self.alpha1 if alpha is None else alpha
    #     beta = self.beta1 if beta is None else beta

    #     f_pc = self.alpha1 + self.beta1 * old_correct
    #     D_pc = torch.mean((output - old_output).pow(2), dim=1) / 2
    #     loss_pc = torch.mean(f_pc * D_pc)
    #     return loss_pc




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
    def __init__(self, old_output_clean=None,
                alpha=0, beta=1):
        super(PCTLoss, self).__init__()
        self.old_output_clean = old_output_clean
        self.alpha = alpha
        self.beta = beta

        self.ce = CrossEntropyLoss()

        keys = ('ce', 'old_dist', 'old_focal')
        self._add_loss_term(keys)
        self.loss_keys = tuple(self.loss_path.keys())


    
    def forward(self, model_output: Tensor, target: Tensor, 
                old_output=None, batch_idx=None, batch_size=None, curr_batch_dim=None) -> Tensor:
        loss_ce = self.ce(model_output, target.long())  # compute cross entropy of the output

        if old_output is None:
            assert batch_idx is not None
            assert batch_size is not None
            assert curr_batch_dim is not None
            # apply a weighting for each training sample based on old model outputs
            old_outs, old_correct = self._logits_to_corrects(self.old_output_clean,
                                                            target, batch_idx,
                                                            batch_size, curr_batch_dim)
        else:
            old_outs = old_output
            preds = torch.argmax(old_outs, dim=1)
            old_correct = (preds == target)

        D_dist = torch.mean((model_output - old_outs).pow(2), dim=1) / 2
        loss_distill = torch.mean(D_dist)

        D_focal = torch.mean((model_output - old_outs).pow(2), dim=1) / 2
        loss_focal = torch.mean(old_correct * D_focal)

        # # combine CE loss and PCT loss
        # loss = (1 - self.alpha - self.beta) * loss_ce + self.alpha * loss_distill + self.beta * loss_focal
        loss = loss_ce + self.alpha * loss_distill + self.beta * loss_focal
        
        
        if self.keep_loss_path:
            self._update_loss_path((loss, loss_ce, loss_distill, loss_focal), self.loss_keys)

        return loss, loss_ce, loss_distill, loss_focal




class MixedPCTLoss(BasePCTLoss):
    def __init__(self, old_output, new_output,
                 alpha=0, beta=1, only_nf=False):
        """

        :param old_output: outputs of old model on the training set
        :param new_output: outputs of the new model from which we finetune
        :param gamma: to deprecate
        :param alpha1: pure distillation, to deprecate
        :param beta1: bonus for NF, only one useful here
        """
        super(MixedPCTLoss, self).__init__()
        self.old_output = old_output
        self.new_output = new_output
        self.alpha = alpha
        self.beta = beta
        self.only_nf = only_nf

        self.ce = CrossEntropyLoss()

        keys = ('ce', 'new_distill', 'old_focal')
        self._add_loss_term(keys)
        self.loss_keys = tuple(self.loss_path.keys())


    def forward(self, model_output: Tensor, target: Tensor, 
                old_output=None, new_output=None, 
                batch_idx=None, batch_size=None, curr_batch_dim=None) -> Tensor:
        loss_ce = self.ce(model_output, target.long())

        if old_output is None:
            # This happen when I already computed the outputs on a given dataset, so I just slice depending on
            # batch index and current dimension
            assert batch_idx is not None
            assert batch_size is not None
            assert curr_batch_dim is not None
            # apply a weighting for each training sample based on old model outputs
            old_outs, old_correct = self._logits_to_corrects(self.old_output, target,
                                                    batch_idx, batch_size, curr_batch_dim)

        else:
            # Otherwise, for example in Adv Training, I have to pass the current output because for a given sample
            # it changes every time depending on the obtained advx of the sample
            old_outs = old_output
            preds = torch.argmax(old_outs, dim=1)
            old_correct = (preds == target)
        
        if new_output is None:
            # Same story but considering the "new" model before the first finetuning iteration
            assert batch_idx is not None
            assert batch_size is not None
            assert curr_batch_dim is not None
            # apply a weighting for each training sample based on old model outputs
            new_outs, new_correct = self._logits_to_corrects(self.new_output, target,
                                                   batch_idx, batch_size, curr_batch_dim)

        else:
            new_outs = new_output
            preds = torch.argmax(new_outs, dim=1)
            new_correct = (preds == target)

        
        
        if self.only_nf:
            old_correct = old_correct.logical_and(new_correct.logical_not())

        # print(f"outs sum: {model_output.sum().item()}")
        # print(f"old_outs sum: {old_outs.sum().item()}")
        # print(f"new_outs sum: {new_outs.sum().item()}")
        
        # Stay near the initial model before finetuning ...
        D_dist = torch.mean((model_output - new_outs).pow(2), dim=1) / 2
        loss_distill = torch.mean(new_correct * D_dist)
        # loss_distill = torch.mean(D_dist)

        # ... while mimicking the reference model where it gets right
        D_focal = torch.mean((model_output - old_outs).pow(2), dim=1) / 2
        loss_focal = torch.mean(old_correct * D_focal)

        # # combine CE loss and PCT loss
        # loss = (1 - self.alpha - self.beta) * loss_ce + self.alpha * loss_distill + self.beta * loss_focal
        loss = loss_ce + self.alpha * loss_distill + self.beta * loss_focal

        if self.keep_loss_path:
            self._update_loss_path((loss, loss_ce, loss_distill, loss_focal), self.loss_keys)

        return loss, loss_ce, loss_distill, loss_focal


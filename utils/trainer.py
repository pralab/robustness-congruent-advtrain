from tqdm import tqdm
import torch
from torch import nn
import numpy as np
from copy import deepcopy
from adv_lib.attacks.auto_pgd import apgd
from random import uniform
from torch.nn import CrossEntropyLoss

def train_epoch(model, device, train_loader, optimizer, epoch, loss_fn):
    model = model.to(device)
    model.train()

    with tqdm(total=len(train_loader)) as t:
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            try:
                loss = loss_fn(output, target)
            except:
                loss = loss_fn(output, target.type(torch.LongTensor).to(device))
            loss.backward()
            optimizer.step()

            t.set_postfix(
                epoch='{}'.format(epoch),
                completed='[{}/{} ({:.0f}%)]'.format(
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader)),
                loss='{:.4f}'.format(loss.item()))
            t.update()
    return

def pc_train_epoch(model, device, train_loader, optimizer, epoch, loss_fn, logger=None):
    if logger is None:
        logger = print
        
    model = model.to(device)
    model.train()
    batch_size = train_loader.batch_size

    # with tqdm(total=len(train_loader)) as t:
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(model_output=output, target=target, old_output=None,
                    batch_idx=batch_idx, batch_size=batch_size, curr_batch_dim=data.shape[0])
        loss[0].backward()
        optimizer.step()

        # loss_paths = {k: v[-1] for (k, v) in loss_fn.loss_path.items()}
        # writer.add_scalar("Loss/train", loss_paths, epoch*len(train_loader)+batch_idx)

        # grad_norm_check = next(model.parameters()).grad.mean()
        # print(f"Grad norm check: {grad_norm_check}")
        # ce_cumul.append(loss[1].item())
        # pc_cumul.append(loss[2].item())

        logger.debug(f"Epoch: {epoch} / Batch: {batch_idx}/{len(train_loader.dataset)} / "\
        f"tot:{loss[0]:.3f}, ce:{loss[1]:.3f}, dist: {loss[2]:.3f}, foc: {loss[3]:.3f}")
        # t.set_postfix(
        #     epoch='{}'.format(epoch),
        #     compl='[{}/{} ({:.0f}%)]'.format(
        #         batch_idx * len(data),
        #         len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader)),
        #     loss='{:.4f}'.format(loss[0].item()))
        # t.update()
    return


def adv_train_epoch(model, device, train_loader,
                    optimizer, epoch, loss_fn,
                    eps=0.03, n_steps=5):
    """
    Set mixmse=True if using MixMSE loss to also keep a copy of the new model before training
    """
    model = model.to(device)
    batch_size = train_loader.batch_size


    with tqdm(total=len(train_loader)) as t:
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # Adv training step
            model.eval()
            advx = apgd(model, data, target,
                        eps=eps, norm=float('inf'), n_iter=n_steps)
            model.train()
            adv_output = model(advx)
            loss = loss_fn(input=adv_output, target=target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t.set_postfix(
                epoch='{}'.format(epoch),
                compl='[{}/{} ({:.0f}%)]'.format(
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader)),
                loss='{:.4f}'.format(loss.item()))
            # loss_ce='{:.4f}'.format(np.array(ce_cumul).mean()),
            # loss_pc='{:.4f}'.format(np.array(pc_cumul).mean()))
            t.update()
    return


def adv_pc_train_epoch(model, old_model, device, train_loader, 
                    optimizer, epoch, loss_fn, new_model=None, mixmse=False,
                    eps=0.03, n_steps=50, logger=None):
    """
    Set mixmse=True if using MixMSE loss to also keep a copy of the new model before training
    """
    if mixmse:
        assert new_model is not None, "MixMSE needs the new reference model as input"
        new_model.eval()
    
    if logger is None:
        logger = print
    
    model = model.to(device)
    model.train()
    old_model.eval()
    batch_size = train_loader.batch_size
    
    # with tqdm(total=len(train_loader)) as t:
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
   
        advx = fgsm_attack(x=data, target=target, model=model, epsilon=eps)
        # advx = data

        # print(f"Params mean: {list(model.parameters())[-1].mean().item()}")
        # print(f"New Params mean: {list(new_model.parameters())[-1].mean().item()}")
        # print(f"Old Params mean: {list(old_model.parameters())[-1].mean().item()}")
        
        adv_output = model(advx)

        with torch.no_grad():
            adv_old_output = old_model(advx)
            if mixmse:
                adv_new_output = new_model(advx)
                    
        if mixmse:
            loss = loss_fn(model_output=adv_output, target=target, 
                                old_output=adv_old_output, new_output=adv_new_output)
        else:
            loss = loss_fn(model_output=adv_output, target=target, old_output=adv_old_output)

        optimizer.zero_grad()
        loss[0].backward()
        optimizer.step()

        logger.debug(f"Epoch: {epoch} / Batch: {batch_idx}/{len(train_loader)} / "\
        f"tot:{loss[0]:.3f}, ce:{loss[1]:.3f}, dist: {loss[2]:.3f}, foc: {loss[3]:.3f}")
    
    return


# def adv_pc_train_epoch(model, old_model, device, train_loader, 
#                     optimizer, epoch, loss_fn, new_model=None, mixmse=False,
#                     eps=0.03, n_steps=50, logger=None):
#     """
#     Set mixmse=True if using MixMSE loss to also keep a copy of the new model before training
#     """
#     if mixmse:
#         assert new_model is not None, "MixMSE needs the new reference model as input"
    
#     if logger is None:
#         logger = print
    
#     model = model.to(device)
#     old_model.eval()
#     batch_size = train_loader.batch_size

#     # with tqdm(total=len(train_loader)) as t:
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)


#         # Implementazione con APGD
#         advx_flag = uniform(0., 1.) < 2
#         if advx_flag:
#             # Set requires_grad attribute of tensor. Important for Attack
#             model.eval()
#             # Adv training step
#             # advx = apgd(model, data, target,
#             #             eps=eps, norm=float('inf'), n_iter=n_steps)    
#             advx = fgsm_attack(data, model, eps)
#             model.train()
#         else:
#             advx = data
#             model.train()
            
#         # print(f"Params mean: {list(model.parameters())[-1].mean().item()}")
#         # print(f"New Params mean: {list(new_model.parameters())[-1].mean().item()}")
#         # print(f"Old Params mean: {list(old_model.parameters())[-1].mean().item()}")
        
#         adv_output = model(advx)

#         with torch.no_grad():
#             adv_old_output = old_model(advx)
#             if mixmse:
#                 adv_new_output = new_model(advx)
                    
#         if mixmse:
#             loss = loss_fn(model_output=adv_output, target=target, 
#                                 old_output=adv_old_output, new_output=adv_new_output)
#         else:
#             loss = loss_fn(model_output=adv_output, target=target, old_output=adv_old_output)

#         optimizer.zero_grad()
#         loss[0].backward()
#         optimizer.step()

#         logger.debug(f"Epoch: {epoch} / Batch: {batch_idx}/{len(train_loader)} / advx: {advx_flag} / "\
#         f"tot:{loss[0]:.3f}, ce:{loss[1]:.3f}, dist: {loss[2]:.3f}, foc: {loss[3]:.3f}")
    
#     return


def freeze_network(model, n_layer=1):
    """
    model: torch network
    n_layer: how many layers you DON'T want to freeze 
    starting from the last one up to the first
    """
    max_id = len(list(model.children())) - n_layer

    for i, child in enumerate(model.children()):
        if i < max_id:
            for param in child.parameters():
                param.requires_grad = False

# FGSM attack code
def fgsm_attack(x, target, model, epsilon):
    x.requires_grad = True
    model.eval()
    
    output = model(x)
    loss = CrossEntropyLoss()(output, target.long())
    loss.backward()
    x_grad = x.grad.data
    sign_x_grad = x_grad.sign()   # Collect the element-wise sign of the data gradient
    
    delta = (torch.rand(x.shape)*2*epsilon - epsilon).to(x.device)
    delta = delta + 1.25*epsilon*sign_x_grad
    delta = torch.clamp(delta, -epsilon, epsilon)
    # Create the perturbed image by adjusting each pixel of the input image
    x_adv = x + delta
    # Adding clipping to maintain [0,1] range
    x_adv = torch.clamp(x_adv, 0, 1)
    
    model.train()
    x.requires_grad = False
    
    # Return the perturbed image

    
    return x_adv








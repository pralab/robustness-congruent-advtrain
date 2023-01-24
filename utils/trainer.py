from tqdm import tqdm
import torch
from torch import nn
import numpy as np
from copy import deepcopy
from adv_lib.attacks.auto_pgd import apgd

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

def pc_train_epoch(model, device, train_loader, optimizer, epoch, loss_fn):
    model = model.to(device)
    model.train()
    batch_size = train_loader.batch_size

    with tqdm(total=len(train_loader)) as t:
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(model_output=output, target=target, old_output=None,
                        batch_idx=batch_idx, batch_size=batch_size, curr_batch_dim=data.shape[0])
            loss[0].backward()
            optimizer.step()

            # ce_cumul.append(loss[1].item())
            # pc_cumul.append(loss[2].item())

            t.set_postfix(
                epoch='{}'.format(epoch),
                compl='[{}/{} ({:.0f}%)]'.format(
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader)),
                loss='{:.4f}'.format(loss[0].item()))
            t.update()
    return


def adv_pc_train_epoch(model, old_model, device, train_loader, 
                    optimizer, epoch, loss_fn, mixmse=False,
                    eps=0.03, n_steps=5):
    """
    Set mixmse=True if using MixMSE loss to also keep a copy of the new model before training
    """
    model = model.to(device)
    old_model.eval()
    batch_size = train_loader.batch_size

    if mixmse:
        new_model = deepcopy(model).to(device)

    with tqdm(total=len(train_loader)) as t:
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # # Standard training step
            # model.train()
            # output_clean = model(data)        
            # loss = loss_fn(model_output=output_clean, target=target,
            #             batch_idx=batch_idx, batch_size=batch_size, curr_batch_dim=data.shape[0])
            # optimizer.zero_grad()
            # loss[0].backward()
            # optimizer.step()

            
            # uniform_noise = torch.rand(data.shape, device=device)

            # noisy_data = data + eps*uniform_noise
            # noisy_output = model(noisy_data)
            # noisy_old_output = old_model(noisy_data)
            # noisy_loss = loss_fn(model_output=noisy_output, target=target, old_output=noisy_old_output)

            model.eval()
            # Adv training step
            advx = apgd(model, data, target,
                        eps=eps, norm=float('inf'), n_iter=n_steps)            
            model.train()
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

            t.set_postfix(
                epoch='{}'.format(epoch),
                compl='[{}/{} ({:.0f}%)]'.format(
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader)),
                loss='{:.4f}'.format(loss[0].item()))
                # loss_ce='{:.4f}'.format(np.array(ce_cumul).mean()),
                # loss_pc='{:.4f}'.format(np.array(pc_cumul).mean()))
            t.update()
    return


def freeze_network(model, n_layer=1):
    max_id = len(list(model.children())) - n_layer

    for i, child in enumerate(model.children()):
        if i < max_id:
            for param in child.parameters():
                param.requires_grad = False








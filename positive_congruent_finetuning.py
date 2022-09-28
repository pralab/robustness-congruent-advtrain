from cmath import exp
from math import ceil
import torchvision.datasets as datasets
from utils.utils import FINETUNING_DIRNAME_DEFAULT, PREDS_DIRNAME_DEFAULT, FT_DEBUG_FOLDER_DEFAULT, \
preds_fname, MODEL_NAMES, custom_dirname, advx_fname
from secml.utils import fm
from robustbench.utils import load_model
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import pandas as pd
import pickle
from utils.eval import predict
from adv_lib.attacks.auto_pgd import apgd

from utils.visualization import show_loss

def finetuning_pipeline(model_names, 
                    x_train, y_train,
                    batch_size, epochs,
                    exp_folder_name, logger, device, adv_training=False, eps=0.03, n_steps=5):

    for model_id, model_name in enumerate(model_names):
        logger.debug(model_names[model_id])
        # ogni modello a partire dal secondo devo finetunarlo con 
        # constraint sui sample classificati correttamente
        # i modelli li salvo su una cartella

        # if model_id != 0:
        models_not_processed = []
        if model_id > 0:
            try:
                finetuning(model_id=model_id, 
                            x_train=x_train, y_train=y_train, 
                            batch_size=batch_size, epochs=epochs, 
                            exp_folder_name=exp_folder_name, logger=logger, device=device,
                            adv_training=adv_training, eps=eps, n_steps=n_steps)
                
                # qui Kang da un errore "underflow in dt nan", qualcosa che succede negli ode
            except Exception as e:
                logger.error(e)
                logger.debug(f"{model_name} not processed.")
                models_not_processed.append(model_name)
    logger.debug(f"Model not processed: {models_not_processed}")
    print("")


def finetuning(model_id, 
    x_train, y_train,
    batch_size, epochs,
    exp_folder_name, logger, device,
    adv_training=False, eps=0.03, n_steps=5):

    model_name = MODEL_NAMES[model_id]
    num_batches = ceil(x_train.shape[0] / batch_size)
    predictions_trset_folder = fm.join(exp_folder_name, custom_dirname(PREDS_DIRNAME_DEFAULT, tr_set=True))
    finetuned_models_folder = fm.join(exp_folder_name, FINETUNING_DIRNAME_DEFAULT)
    if not fm.folder_exist(finetuned_models_folder):
        fm.make_folder(finetuned_models_folder)
    
    ft_debug_folder = fm.join(exp_folder_name, FT_DEBUG_FOLDER_DEFAULT)
    if not fm.folder_exist(ft_debug_folder):
        fm.make_folder(ft_debug_folder)

    logger.debug('Loading old model predictions')
    with open(fm.join(predictions_trset_folder, f"{model_name}.gz"), 'rb') as f:
        # prendo advs[0] perchè sto usando un solo epsilon
        data = pickle.load(f)
    old_preds_clean, old_outputs_clean = torch.tensor(data['preds']), data['outs'].to(device)

    with open(fm.join(predictions_trset_folder, advx_fname(model_name)), 'rb') as f:
        data = pickle.load(f)
    old_preds_adv, old_outputs_adv = torch.tensor(data['preds']), data['outs'].to(device)

    # maschera booleana dove ho 1 sui sample classificati correttamente
    # maschera adv booleana dove ho 1 sui sample che anche da adv 
    # vengono classificati correttamente, e su questo ci sarà da fare adv training probably

    model = load_model(model_name=model_name, dataset='cifar10', threat_model='Linf')
    model.train()
    model.to(device)
    freeze_network(model)

    debug_df = pd.DataFrame(columns=['epoch', 'batch', 'L', 'CE', 'PC', 'RPC'])
    debug_df.index.name = 'iter'

    iter = 0
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(epochs):
        running_loss = 0.0
        running_loss_ce = 0.0
        running_loss_pc = 0.0
        running_loss_rpc = 0.0
        for batch_i in range(num_batches):
            start_i = batch_i * batch_size
            end_i = start_i + batch_size
            x_i, y_i = x_train[start_i:end_i], y_train[start_i:end_i]
            x_i = x_i.to(device)
            y_i = y_i.to(device)
            # old_preds_mask_i = old_preds_mask[start_i:end_i]

            # if adv_training and (batch_i%2 == 0):
            #     model.eval()
            #     advx_i = apgd(model, x_i, y_i,
            #                     eps=eps, norm=float('inf'), n_iter=n_steps)
            #     model.train()

            optimizer.zero_grad()

            out = model(x_i)
            loss, loss_ce, loss_pc, loss_rpc = custom_loss(out, y_i, 
                                                 old_outputs_clean[start_i:end_i],
                                                 old_outputs_adv[start_i:end_i])
            debug_df.loc[iter] = [epoch, batch_i,
                                loss.item(), 
                                loss_ce.item(), 
                                loss_pc.item(), 
                                loss_rpc.item()]
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            running_loss_ce += loss_ce.item()
            running_loss_pc += loss_pc.item()
            running_loss_rpc += loss_rpc.item()
            iter += 1
        #     print(f'[Epoch: {epoch + 1}] Loss: {loss.item():.3f} \
        #     / CE: {running_ce_loss/loss_ce.item():.3f} \
        #     / PC: {loss_pc.item():.3f}')

        # print(f'[Epoch: {epoch + 1}] Loss: {running_loss/num_batches:.3f}')
        # running_loss = 0.0
        logger.debug(f"[Epoch: {epoch + 1}] Loss: {running_loss/num_batches:.3f} /"
        f" CE: {running_loss_ce/num_batches:.3f}"
        f"/ PC: {running_loss_pc/num_batches:.3f}"
        f"/ RPC: {running_loss_rpc/num_batches:.3f}")
        running_loss = 0.0
        running_loss_ce = 0.0
        running_loss_pc = 0.0
        running_loss_rpc = 0.0
        print("")
    
    path = fm.join(finetuned_models_folder, f"{model_name}.pt")
    torch.save(model.state_dict(), path)
    debug_df.to_csv(fm.join(ft_debug_folder, f"{model_name}.csv"))


    # show_loss(ft_debug_folder, model_name)
    # preds_orig = predict(old_model, x_test, batch_size, device)
    # model2 = load_model(model_name=model_name, dataset='cifar10', threat_model='Linf')
    # preds_new = predict(model2, x_test, batch_size, device)
    # model2.load_state_dict(torch.load(path))
    # preds_new_ftuned = predict(model2, x_test, batch_size, device)


    

def custom_loss(output, target, old_output_clean, old_output_rob, 
                gamma1=1, alpha1=0, beta1=1,
                gamma2=1, alpha2=0, beta2=1):
    # cross entropy loss to minimize classification error
    loss_ce = F.cross_entropy(output, target, reduction='none')
    
    # apply a weighting for each training sample based on old model outputs
    old_preds_clean = torch.argmax(old_output_clean, dim=1)
    f = alpha1 + beta1*((target == old_preds_clean).float()) 
    # distance bw new and old model outputs
    D = F.mse_loss(output, old_output_clean, reduction='none').sum(dim=1)
    # weight differently samples classified correctly by the old model
    loss_pc = f*D

    # apply a weighting for each training sample based on old model outputs
    old_preds_rob = torch.argmax(old_output_rob, dim=1)
    f = alpha2 + beta2*((target == old_preds_rob).float()) 
    # distance bw new and old model outputs
    D = F.mse_loss(output, old_output_rob, reduction='none').sum(dim=1)
    # weight differently samples classified correctly by the old model
    loss_rpc = f*D

    # combine CE loss and PCT loss
    loss = (loss_ce + gamma1*loss_pc + gamma2*loss_rpc).mean()

    return loss, loss_ce.mean(), loss_pc.mean(), loss_rpc.mean()


def freeze_network(model):
    max_id = len(list(model.children())) - 1

    for i, child in enumerate(model.children()):
        if i != max_id:
            for param in child.parameters():
                param.requires_grad = False
    
    # return model

if __name__ == '__main__':

    device2 = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    for model_name in MODEL_NAMES:
        model = load_model(model_name=model_name, dataset='cifar10', threat_model='Linf')
        model.train()
        model.to(device2)        

        print(model_name)
        
        c=0
        for param in model.parameters():
            if param.requires_grad:
                c+=1
        print(c)

        max_id = len(list(model.children())) - 1

        freeze_network(model)

        c=0
        for param in model.parameters():
            if param.requires_grad:
                c+=1
        print(c)
        print("")

    print("")

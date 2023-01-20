import torch
from utils.utils import set_all_seed, MODEL_NAMES
from utils.data import split_train_valid, get_cifar10_dataset
from utils.eval import correct_predictions
from torch.utils.data import DataLoader
from robustbench.utils import load_model
import os
import pickle

if __name__ == '__main__':
    batch_size = 200
    n_steps = 50
    n_ts = None
    device = 'cuda:1'

    models_id = (0,1,2,3,4,5,6,7)

    #####################################
    # PREPARE DATA
    #####################################
    test_dataset = get_cifar10_dataset(train=False, shuffle=False, num_samples=n_ts)
    # shuffle can be set to True if reference models are evaluated on the fly
    # without exploiting precomputed outputs
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for model_id in models_id:
        model_name = MODEL_NAMES[model_id] 
        model = load_model(model_name=model_name, dataset='cifar10', threat_model='Linf')
        # each model has a folder in wich I save the correct predictions
        base_path = os.path.join('results', 'clean', model_name)
        if not os.path.isdir(base_path):
            os.makedirs(base_path)
        
        corrects_path = os.path.join(base_path, 'correct_preds.gz')

        correct_preds = correct_predictions(model=model, test_loader=test_loader, device=device)
        with open(corrects_path, 'wb') as f:
             pickle.dump(correct_preds.cpu(), f)
        


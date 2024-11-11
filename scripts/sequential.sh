export PYTHONPATH='../':$PYTHONPATH
#!/bin/bash
echo "PWD: $PWD"

cuda="0"
echo "Inserted cuda $cuda"

echo "PID of this script: $$"

dataset="imagenet"
n_tr="50000"
n_ts="10000"
lr="1e-3"
optim="sgd"

# # epochs and batch_size for M1,2,3
# epochs="10"
# batch_size="128"
# epochs and batch_size for M4
epochs="5"
batch_size="64"

n_steps="50"
n_adv_ts="10000"
# n_tr="500"
# n_ts="100"
# epochs="1"
# lr="1e-3"
# batch_size="500"
# n_steps="5"
# n_adv_ts="20"

# All 14 combinations of models with both increasing clean and robust accuracy


# old_model_ids="0 1 2"
# model_ids="1 2 3"
old_model_ids="2 3"
model_ids="3 4"

# if test "$cuda" = "0"
# then
#     old_model_ids="1 1 2 2 2 3 3"
#     model_ids="4 7 4 5 7 2 4"
# else
#     old_model_ids="3 3 3 4 5 5 6"
#     model_ids="5 6 7 7 4 7 7"
# fi
# old_model_ids="1 1 2 2 2 3 3"
# model_ids="4 7 4 5 7 2 4"
# old_model_ids="3 3 3 4 5 5 6"
# model_ids="5 6 7 7 4 7 7"
# old_model_ids="1 1 2 2 2 3 3 3 3 3 4 5 5 6"
# model_ids="4 7 4 5 7 2 4 5 6 7 7 4 7 7"


loss_names="MixMSE-AT PCT-AT PCT"
# alphas="0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0"
# betas="0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9" 

alphas_pct="1"
betas_pct="2"   

alphas_mix="0.5"
betas_mix="0.4" 
# alphas_mix="0.75"
# betas_mix="0.2"   

# quelli con alpha=1 li tolgo perchè peggiora troppo robustness
# noi abbiamo aggiunto normalizzazione
# il nostro loss è solo quello filtrato moltiplicato per distillation

root="results"

# exp_name="SPERANZA_2024"
# exp_name="IMAGENET_SGD_LR-1e-2_30epochs"
# exp_name="day-26-09-2024_hr-17-34-46_IMAGENET_FIRST_TRIAL"
exp_name="SEQUENTIAL_FINETUNING"

cmd="nohup python -m train_model -dataset $dataset -n_tr $n_tr -n_ts $n_ts -epochs $epochs -lr $lr -optim $optim -batch_size $batch_size -n_steps $n_steps -n_adv_ts $n_adv_ts -old_model_ids $old_model_ids -model_ids $model_ids -loss_names $loss_names -alphas_pct $alphas_pct -betas_pct $betas_pct -alphas_mix $alphas_mix -betas_mix $betas_mix -root $root -cuda $cuda" 
# $cmd -exp_name $exp_name_at -adv_tr &> results/nohup_${exp_name_at}.out; 
# $cmd -exp_name $exp_name -date &> nohups/nohup_${exp_name}.out &

# $cmd -exp_name $exp_name -date &> nohups/nohup_${exp_name}.out &
$cmd -exp_name $exp_name -sequential -skip_first &> nohups/nohup_${exp_name}.out &

# Inserted cuda 0
# PID of this script: 525673


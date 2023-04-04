# export PYTHONPATH='../':$PYTHONPATH


# alphas="0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0"
# betas="0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"

n_tr="50000"
n_ts="10000"
epochs="12"
lr="1e-3"
batch_size="500"
n_steps="50"
n_adv_ts="2000"

# All 14 combinations of models with both increasing clean and robust accuracy
old_model_ids="1 1 2 2 2 3 3 3 3 3 4 5 5 6"
model_ids="4 7 4 5 7 2 4 5 6 7 7 4 7 7"

# old_model_ids="1"
# model_ids="4"

loss_names="MixMSE-AT PCT-AT"
# alphas="0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0"
# betas="0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"
alphas_mix="0.75 0.7 0.5 0.3"
betas_mix="0.2 0.2 0.4 0.6"
alphas_pct="1 1 1 1"
betas_pct="1 2 5 10"

root="results"
exp_name="PIPELINE_50k_3models_31March"
exp_path="day-30-03-2023_hr-10-01-01_PIPELINE_50k_3models"


# #DEBUG
# n_tr="20"
# n_ts="20"
# epochs="2"
# lr="1e-3"
# batch_size="10"
# n_steps="1"
# n_adv_ts="10"

# # All 14 combinations of models with both increasing clean and robust accuracy
# old_model_ids="2 3"
# model_ids="5 6"

# # old_model_ids="1"
# # model_ids="4"

# loss_names="MixMSE-AT PCT-AT"
# # alphas="0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0"
# # betas="0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"
# alphas_mix="0.75 0.7"
# betas_mix="0.2 0.2"
# alphas_pct="1 1"
# betas_pct="1 2"
# root="results"
# cuda="0"
# exp_name="DEBUG"
# exp_path="DEBUG"

cmd="nohup python -m train_model -n_tr $n_tr -n_ts $n_ts -epochs $epochs -lr $lr -batch_size $batch_size -n_steps $n_steps -n_adv_ts $n_adv_ts -old_model_ids $old_model_ids -model_ids $model_ids -loss_names $loss_names -alphas_pct $alphas_pct -betas_pct $betas_pct -alphas_mix $alphas_mix -betas_mix $betas_mix -root $root" 

# $cmd -exp_name $exp_name_at -adv_tr &> results/nohup_${exp_name_at}.out; 
$cmd -exp_name $exp_path &> results/nohup_${exp_name}.out 

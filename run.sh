# n_tr="50000"
# n_ts="10000"
# epochs="12"
# lr="1e-3"
# batch_size="500"
# n_steps="50"
# n_adv_ts="2000"
# old_model_ids="1"
# model_ids="4"
# loss_names="MixMSE"
# alphas="0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0"
# betas="0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"
# cuda="0"

#DEBUG
n_tr="50000"
n_ts="10000"
epochs="12"
lr="1e-3"
batch_size="500"
n_steps="50"
n_adv_ts="2000"

old_model_ids="1 2 3"
model_ids="4 5 6"
loss_names="PCT"
# alphas="0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0"
# betas="0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"
alphas_mix="0.7 0.5 0.3"
betas_mix="0.2 0.4 0.6"
alphas_pct="1 1 1 1"
betas_pct="1 2 5 10"

cuda="0"

# -date
cmd="nohup python -m train_model -n_tr $n_tr -n_ts $n_ts -epochs $epochs -lr $lr -batch_size $batch_size -n_steps $n_steps -n_adv_ts $n_adv_ts -old_model_ids $old_model_ids -model_ids $model_ids -loss_names $loss_names -alphas_pct $alphas_pct -betas_pct $betas_pct -alphas_mix $alphas_mix -betas_mix $betas_mix" 

# exp_name="day-27-03-2023_hr-19-59-52_CHECK_PCT_MIXMSE"
# $cmd -exp_name WTF_MIXMSE_PCT   # && 

# exp_name="PIPELINE_PCT_MIXMSE"
# exp_name_at="${exp_name}_AT"

# $cmd -exp_name $exp_name_at -adv_tr &> results/nohup_${exp_name_at}.out; 
# $cmd -exp_name $exp_name &> results/nohup_${exp_name}.out 



exp_name="day-25-01-2023_hr-15-38-00_CLEAN_TR"
$cmd -exp_name $exp_name -test_only &> results/nohup_repeat_eval.out 
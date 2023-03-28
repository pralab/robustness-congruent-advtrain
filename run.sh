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
old_model_ids="1"
model_ids="4"
loss_names="PCT MixMSE"
# alphas="0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0"
# betas="0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"
alphas="0.7"
betas="0.2"
cuda="0"


cmd="nohup python -m train_model -date -n_tr $n_tr -n_ts $n_ts -epochs $epochs -lr $lr -batch_size $batch_size -n_steps $n_steps -n_adv_ts $n_adv_ts -old_model_ids $old_model_ids -model_ids $model_ids -loss_names $loss_names -alphas $alphas -betas $betas" 

# $cmd -exp_name WTF_MIXMSE_PCT && 
$cmd -exp_name WTF_MIXMSE_PCT_AT -adv_tr
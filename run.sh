pid=47164

# An entry in /proc means that the process is still running.
while [ -d "/proc/$pid" ]; do
    sleep 60
done

cmd="nohup python -m train_model -cuda 0 -adv_tr 1 -exp_name ADV_TR &> results/nohup_advtr2.out &"

$cmd



###################################################################################

# N_EXAMPLES="2000"
# N_TR_EXAMPLES="5000"
# #Attack constants
# EPSILON="0.03"
# N_STEPS="250"
# #Predictions constants
# BATCH_SIZE="200"
# ROOT="data"
# LR='1e-4'
# EPOCHS='100'
# GAMMA1='1'
# GAMMA2='0'

# EXP_FT_NAME="tssample-${N_EXAMPLES}_trsamples-${N_TRAIN_SAMPLES}_batchsize-${BATCH_SIZE}_lr-${LR}_g1-${GAMMA1}_g2-${GAMMA2}"

# # cmd="nohup python main_pipeline.py -n_examples $N_EXAMPLES -n_tr_examples $N_TR_EXAMPLES -eps $EPSILON -n_steps $N_STEPS -batch_size $BATCH_SIZE"
# cmd="nohup python main_pipeline.py -n_examples $N_EXAMPLES -n_tr_examples $N_TR_EXAMPLES -batch_size $BATCH_SIZE -exp_ft_name $EXP_FT_NAME -gamma1 $GAMMA1 -gamma2 $GAMMA2"

# $cmd

# ###################################################################################

# N_EXAMPLES="2000"
# N_TR_EXAMPLES="5000"
# #Attack constants
# EPSILON="0.03"
# N_STEPS="250"
# #Predictions constants
# BATCH_SIZE="1000"
# ROOT="data"
# LR='1e-4'
# EPOCHS='200'
# GAMMA1='1'
# GAMMA2='0'
# CUDA='0'

# # EXP_FT_NAME="tssample-${N_EXAMPLES}_trsamples-${N_TRAIN_SAMPLES}_batchsize-${BATCH_SIZE}_lr-${LR}_g1-${GAMMA1}_g2-${GAMMA2}"

# # cmd="nohup python main_pipeline.py -n_examples $N_EXAMPLES -n_tr_examples $N_TR_EXAMPLES -eps $EPSILON -n_steps $N_STEPS -batch_size $BATCH_SIZE -gamma2 $GAMMA2"
# cmd="nohup python main_pipeline.py -n_examples $N_EXAMPLES -n_tr_examples $N_TR_EXAMPLES -batch_size $BATCH_SIZE -epochs $EPOCHS -gamma2 $GAMMA2"
# # EXP_FT_NAME="tssample-${N_EXAMPLES}_trsamples-${N_TR_EXAMPLES}_batchsize-${BATCH_SIZE}_g1-${GAMMA1}_g2-${GAMMA2}"


# lr1=1
# lr2=5e-1
# exp_ft_name_1="gamma1-${lr1}"
# exp_ft_name_2="gamma1-${lr2}"
# $cmd -gamma1 $lr1 -exp_ft_name $exp_ft_name_1 -cuda_id 0 & 
# $cmd -gamma1 $lr2 -exp_ft_name $exp_ft_name_2 -cuda_id 1 &
# wait

# lr1=1e-1
# lr2=5e-2
# exp_ft_name_1="gamma1-${lr1}"
# exp_ft_name_2="gamma1-${lr2}"
# $cmd -gamma1 $lr1 -exp_ft_name $exp_ft_name_1 -cuda_id 0 & 
# $cmd -gamma1 $lr2 -exp_ft_name $exp_ft_name_2 -cuda_id 1 &
# wait

# lr1=1e-2
# lr2=5e-3
# exp_ft_name_1="gamma1-${lr1}"
# exp_ft_name_2="gamma1-${lr2}"
# $cmd -gamma1 $lr1 -exp_ft_name $exp_ft_name_1 -cuda_id 0 & 
# $cmd -gamma1 $lr2 -exp_ft_name $exp_ft_name_2 -cuda_id 1 &
# wait

# lr1=1e-3
# lr2=5e-4
# exp_ft_name_1="gamma1-${lr1}"
# exp_ft_name_2="gamma1-${lr2}"
# $cmd -gamma1 $lr1 -exp_ft_name $exp_ft_name_1 -cuda_id 0 & 
# $cmd -gamma1 $lr2 -exp_ft_name $exp_ft_name_2 -cuda_id 1 &
# wait

# lr1=1e-4
# lr2=5e-5
# exp_ft_name_1="gamma1-${lr1}"
# exp_ft_name_2="gamma1-${lr2}"
# $cmd -gamma1 $lr1 -exp_ft_name $exp_ft_name_1 -cuda_id 0 & 
# $cmd -gamma1 $lr2 -exp_ft_name $exp_ft_name_2 -cuda_id 1 &
# wait


# $cmd -lr 1e-2 -cuda_id 0 & $cmd -lr 5e-3 -cuda_id 1
# wait
# $cmd -lr 1e-3 -cuda_id 0 & $cmd -lr 5e-4 -cuda_id 1
# wait

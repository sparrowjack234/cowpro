#!/bin/bash
# train a model to segment abdominal MRI 
GPUID1=0
export CUDA_VISIBLE_DEVICES=$GPUID1

####### Shared configs ######
PROTO_GRID=8 # using 32 / 8 = 4, 4-by-4 prototype pooling window during training
CPT="myexp"
DATASET='CHAOST2_Superpix'
NWORKER=4
ALL_EV=(  0)
ALL_SCALE=( "MIDDLE" ) # config of pseudolabels

### Use L/R kidney as testing classes
LABEL_SETS=0 
EXCLU='[2,3]' # setting 2: excluding kidneies in training set to test generalization capability even though they are unlabeled. Use [] for setting 1 by Roy et al.

# Comment block for alternative settings - correct bash comment format:


###### Training configs (irrelavent in testing) ######
NSTEP=100100
DECAY=0.95
MAX_ITER=1000 # defines the size of an epoch
SNAPSHOT_INTERVAL=25000 # interval for saving snapshot
SEED='1234'

###### Validation configs ######
SUPP_ID='[4]'  # using the additionally loaded scan as support

echo ===================================
for EVAL_FOLD in "${ALL_EV[@]}"
do
    for SUPERPIX_SCALE in "${ALL_SCALE[@]}"
    do
    PREFIX="test_vfold${EVAL_FOLD}"
    echo $PREFIX
    LOGDIR="./exps/${CPT}"
    if [ ! -d $LOGDIR ]
    then
        mkdir $LOGDIR
    fi
    RELOAD_PATH='E:\Suyash\cowpro\runs\myexperiments_MIDDLE_0_opencv_grabcut\mySSL_train_CHAOST2_Superpix_lbgroup0_scale_MIDDLE_vfold0_CHAOST2_Superpix_sets_0_1shot\3\snapshots/100000.pth' # path to the reloaded model
    
    # Run validation without parameter tuning flags
    python validation_iter.py with \
    'modelname=dlfcn_res101' \
    'usealign=True' \
    'optim_type=sgd' \
    reload_model_path=$RELOAD_PATH \
    num_workers=$NWORKER \
    scan_per_load=-1 \
    label_sets=$LABEL_SETS \
    'use_wce=True' \
    exp_prefix=$PREFIX \
    'clsname=grid_proto' \
    n_steps=$NSTEP \
    exclude_cls_list=$EXCLU \
    eval_fold=$EVAL_FOLD \
    dataset=$DATASET \
    proto_grid_size=$PROTO_GRID \
    max_iters_per_load=$MAX_ITER \
    min_fg_data=1 seed=$SEED \
    save_snapshot_every=$SNAPSHOT_INTERVAL \
    superpix_scale=$SUPERPIX_SCALE \
    lr_step_gamma=$DECAY \
    path.log_dir=$LOGDIR \
    support_idx=$SUPP_ID
    done
done
#!/bin/bash
# Validation script for supervised CoWPro model - Manual path specification
GPUID1=0
export CUDA_VISIBLE_DEVICES=$GPUID1

####### Shared configs ######
PROTO_GRID=8 # using 32 / 8 = 4, 4-by-4 prototype pooling window during training
CPT="supervised_cowpro_test"
DATASET='CHAOST2'
NWORKER=0

ALL_EV=(0 ) # 5-fold cross validation (0, 1, 2, 3, 4)

### Use L/R kidney as testing classes
LABEL_SETS=0 
EXCLU='[2,3]' # setting 2: excluding kidneys in training set to test generalization capability

### Use Liver and spleen as testing classes
# LABEL_SETS=1 
# EXCLU='[1,4]' 

###### Training configs (irrelevant in testing) ######
NSTEP=50000
DECAY=0.95

MAX_ITER=1000 # defines the size of an epoch
SNAPSHOT_INTERVAL=10000 # interval for saving snapshot
SEED='1234'

###### Validation configs ######
SUPP_ID='[0,1,2,3,4]' # using the additionally loaded scan as support

echo ===================================

for EVAL_FOLD in "${ALL_EV[@]}"
do
    PREFIX="test_supervised_vfold${EVAL_FOLD}"
    echo $PREFIX
    LOGDIR="./exps_test_supervised/${CPT}_${LABEL_SETS}"

    if [ ! -d $LOGDIR ]
    then
        mkdir -p $LOGDIR
    fi

    # ===================================================================
    # MANUAL PATH SPECIFICATION - UPDATE THESE PATHS FOR EACH FOLD
    # ===================================================================
    
    case $EVAL_FOLD in
        0)
            # FOLD 0 - Update this path to your trained model
            RELOAD_PATH="E:\Suyash\iterative refinement\cowpro_supervised\exps_supervised\supervised_cowpro_0\mySSL_supervised_train_CHAOST2_Supervised_lbgroup0_vfold0_mindist4_CHAOST2_Supervised_sets_0_1shot\10\snapshots/50000.pth"
            ;;
        1)
            # FOLD 1 - Update this path to your trained model
            RELOAD_PATH="./exps_supervised/supervised_cowpro_0/mySSL_supervised_train_CHAOST2_Supervised_lbgroup0_vfold1_mindist4_CHAOST2_Supervised_sets_0_1shot/1/snapshots/50000.pth"
            ;;
        2)
            # FOLD 2 - Update this path to your trained model
            RELOAD_PATH="./exps_supervised/supervised_cowpro_0/mySSL_supervised_train_CHAOST2_Supervised_lbgroup0_vfold2_mindist4_CHAOST2_Supervised_sets_0_1shot/1/snapshots/50000.pth"
            ;;
        3)
            # FOLD 3 - Update this path to your trained model
            RELOAD_PATH="./exps_supervised/supervised_cowpro_0/mySSL_supervised_train_CHAOST2_Supervised_lbgroup0_vfold3_mindist4_CHAOST2_Supervised_sets_0_1shot/1/snapshots/50000.pth"
            ;;
        4)
            # FOLD 4 - Update this path to your trained model
            RELOAD_PATH="./exps_supervised/supervised_cowpro_0/mySSL_supervised_train_CHAOST2_Supervised_lbgroup0_vfold4_mindist4_CHAOST2_Supervised_sets_0_1shot/1/snapshots/50000.pth"
            ;;
        *)
            echo "Unknown fold: $EVAL_FOLD"
            continue
            ;;
    esac
    
    # ===================================================================
    # CHECK IF MODEL EXISTS
    # ===================================================================
    
    if [ ! -f "$RELOAD_PATH" ]; then
        echo "❌ Model file not found for fold $EVAL_FOLD: $RELOAD_PATH"
        echo "Please update the path in the script for fold $EVAL_FOLD"
        echo ""
        continue
    fi
    
    echo "✅ Found model for fold $EVAL_FOLD: $RELOAD_PATH"
    
    # ===================================================================
    # RUN VALIDATION
    # ===================================================================
    
    python validation_supervised.py with \
    'modelname=dlfcn_res101' \
    'usealign=True' \
    'optim_type=sgd' \
    reload_model_path="$RELOAD_PATH" \
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
    min_fg_data=1 \
    seed=$SEED \
    save_snapshot_every=$SNAPSHOT_INTERVAL \
    lr_step_gamma=$DECAY \
    path.log_dir=$LOGDIR \
    support_idx=$SUPP_ID \
    val_wsize=2 \
    z_margin=0

    VALIDATION_EXIT_CODE=$?

    if [ $VALIDATION_EXIT_CODE -eq 0 ]; then
        echo "✅ Validation completed successfully for fold $EVAL_FOLD"
    else
        echo "❌ Validation failed for fold $EVAL_FOLD (exit code: $VALIDATION_EXIT_CODE)"
    fi
    
    echo "========================================="
done

echo ""
echo "Validation completed for all folds!"
echo ""
echo "📁 Results saved in: $LOGDIR"
echo ""
echo "📊 To analyze results, run:"
echo "python extract_results.py --results_dir $LOGDIR --output validation_summary.csv --class_analysis"
echo ""
echo "🔍 Manual path verification:"
echo "Check that each fold's model path exists before running:"
echo ""
echo "Fold 0: ./exps_supervised/supervised_cowpro_0/mySSL_*_vfold0_*/*/snapshots/50000.pth"
echo "Fold 1: ./exps_supervised/supervised_cowpro_0/mySSL_*_vfold1_*/*/snapshots/50000.pth"
echo "Fold 2: ./exps_supervised/supervised_cowpro_0/mySSL_*_vfold2_*/*/snapshots/50000.pth"
echo "Fold 3: ./exps_supervised/supervised_cowpro_0/mySSL_*_vfold3_*/*/snapshots/50000.pth"
echo "Fold 4: ./exps_supervised/supervised_cowpro_0/mySSL_*_vfold4_*/*/snapshots/50000.pth"
echo ""
echo "💡 To find your actual paths, use:"
echo "find ./exps_supervised -name '*.pth' -path '*/snapshots/*' | sort"

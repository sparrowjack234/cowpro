#!/bin/bash
# Training script for Cross-Attention Enhanced CoWPro Model
GPUID1=0
export CUDA_VISIBLE_DEVICES=$GPUID1

####### Shared configs ######
PROTO_GRID=8 # using 32 / 8 = 4, 4-by-4 prototype pooling window during training
CPT="cross_attn_cowpro"
DATASET='CHAOST2_Supervised'  # or 'SABS_Supervised'
NWORKER=0

ALL_EV=(0 ) # 5-fold cross validation
MIN_SLICE_DIST=4    # Minimum distance between support and query slices
MAX_DIST_RATIO=0.1667  # 1/6 as decimal

### Use L/R kidney as testing classes
LABEL_SETS=0 
EXCLU='[2,3]' # setting 2: excluding kidneys in training set to test generalization

### Use Liver and spleen as testing classes
# LABEL_SETS=1 
# EXCLU='[1,4]' 

####### Cross-Attention specific configs ######
USE_CROSS_ATTN=True         # Enable cross-attention
CROSS_ATTN_HEADS=8          # Number of attention heads
CROSS_ATTN_LAYERS=2         # Number of cross-attention layers
CROSS_ATTN_DROPOUT=0.1      # Cross-attention dropout rate

###### Training configs ######
NSTEP=50000  # Training steps for supervised learning
DECAY=0.95

MAX_ITER=1000 # defines the size of an epoch
SNAPSHOT_INTERVAL=10000 # interval for saving snapshot
SEED='1234'

echo ===================================
echo "Training Cross-Attention Enhanced CoWPro Model"
echo "Cross-Attention: $USE_CROSS_ATTN"
echo "Attention Heads: $CROSS_ATTN_HEADS"
echo "Attention Layers: $CROSS_ATTN_LAYERS"
echo ===================================

for EVAL_FOLD in "${ALL_EV[@]}"
do
    PREFIX="cross_attn_train_${DATASET}_lbgroup${LABEL_SETS}_vfold${EVAL_FOLD}_heads${CROSS_ATTN_HEADS}_layers${CROSS_ATTN_LAYERS}"
    echo $PREFIX
    LOGDIR="./exps_cross_attn/${CPT}_${LABEL_SETS}"

    if [ ! -d $LOGDIR ]
    then
        mkdir -p $LOGDIR
    fi

    echo "Starting training for fold $EVAL_FOLD with cross-attention..."

    python training_supervised_cross_attn.py with \
    'modelname=dlfcn_res101' \
    'usealign=True' \
    'optim_type=sgd' \
    num_workers=$NWORKER \
    scan_per_load=-1 \
    label_sets=$LABEL_SETS \
    'use_wce=True' \
    'use_tversky=True' \
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
    min_slice_distance=$MIN_SLICE_DIST \
    max_distance_ratio=$MAX_DIST_RATIO \
    which_aug='aug_v3' \
    print_interval=200 \
    model.use_cross_attention=$USE_CROSS_ATTN \
    model.cross_attn_heads=$CROSS_ATTN_HEADS \
    model.cross_attn_layers=$CROSS_ATTN_LAYERS \
    model.cross_attn_dropout=$CROSS_ATTN_DROPOUT

    TRAINING_EXIT_CODE=$?

    if [ $TRAINING_EXIT_CODE -eq 0 ]; then
        echo "✅ Training completed successfully for fold $EVAL_FOLD"
        echo "📁 Model saved in: $LOGDIR/$PREFIX"
        echo "📊 Training logs and visualizations available in experiment directory"
        echo "🎯 Attention maps and analysis saved in: $LOGDIR/$PREFIX/*/attention_maps/"
    else
        echo "❌ Training failed for fold $EVAL_FOLD (exit code: $TRAINING_EXIT_CODE)"
        echo "Check logs for details"
    fi
    
    echo "========================================="
done

echo ""
echo "🎉 Training completed for all folds!"
echo ""
echo "📁 All results saved in: $LOGDIR"
echo ""
echo "📋 Summary of trained models:"
for EVAL_FOLD in "${ALL_EV[@]}"
do
    PREFIX="cross_attn_train_${DATASET}_lbgroup${LABEL_SETS}_vfold${EVAL_FOLD}_heads${CROSS_ATTN_HEADS}_layers${CROSS_ATTN_LAYERS}"
    MODEL_PATH="$LOGDIR/$PREFIX/*/snapshots/"
    echo "  Fold $EVAL_FOLD: $MODEL_PATH"
done
echo ""
echo "🔍 To find exact model paths, run:"
echo "find $LOGDIR -name '*.pth' -path '*/snapshots/*' | sort"
echo ""
echo "📊 Cross-Attention Analysis:"
echo "- Attention maps saved during training for visualization"
echo "- Enhanced training snapshots with attention visualization"
echo "- Detailed attention analysis data for research"
echo ""
echo "🚀 Next step: Run validation with test_cross_attn_supervised_abdomen_mri.sh"

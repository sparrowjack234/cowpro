#!/bin/bash
# Validation script for Cross-Attention Enhanced CoWPro Model
GPUID1=0
export CUDA_VISIBLE_DEVICES=$GPUID1

####### Shared configs ######
PROTO_GRID=8 # using 32 / 8 = 4, 4-by-4 prototype pooling window during training
CPT="cross_attn_cowpro_test"
DATASET='CHAOST2'  # Base dataset name (without _Supervised suffix for validation)
NWORKER=0

ALL_EV=(0 ) # 5-fold cross validation

### Use L/R kidney as testing classes
LABEL_SETS=0 
EXCLU='[2,3]' # setting 2: excluding kidneys in training set to test generalization capability

### Use Liver and spleen as testing classes
# LABEL_SETS=1 
# EXCLU='[1,4]' 

####### Cross-Attention specific configs ######
USE_CROSS_ATTN=True         # Enable cross-attention
CROSS_ATTN_HEADS=8          # Number of attention heads (must match training)
CROSS_ATTN_LAYERS=2         # Number of cross-attention layers (must match training)
CROSS_ATTN_DROPOUT=0.1      # Cross-attention dropout rate

###### Training configs (irrelevant in testing but needed for compatibility) ######
NSTEP=50000
DECAY=0.95
MAX_ITER=1000
SNAPSHOT_INTERVAL=10000
SEED='1234'

###### Validation configs ######
SUPP_ID='[0,1,2,3,4]' # using multiple support scans for robust evaluation

echo ===================================
echo "Validating Cross-Attention Enhanced CoWPro Model"
echo "Cross-Attention: $USE_CROSS_ATTN"
echo "Attention Heads: $CROSS_ATTN_HEADS"
echo "Attention Layers: $CROSS_ATTN_LAYERS"
echo ===================================

for EVAL_FOLD in "${ALL_EV[@]}"
do
    PREFIX="test_cross_attn_vfold${EVAL_FOLD}_heads${CROSS_ATTN_HEADS}"
    echo $PREFIX
    LOGDIR="./exps_test_cross_attn/${CPT}_${LABEL_SETS}"

    if [ ! -d $LOGDIR ]
    then
        mkdir -p $LOGDIR
    fi

    # ===================================================================
    # AUTOMATIC PATH DETECTION FOR TRAINED MODELS
    # ===================================================================
    
    # Try to find the trained model automatically
    SEARCH_PATTERN="./exps_cross_attn/cross_attn_cowpro_${LABEL_SETS}/cross_attn_train_*_vfold${EVAL_FOLD}_heads${CROSS_ATTN_HEADS}_layers${CROSS_ATTN_LAYERS}_*/*/snapshots/*.pth"
    
    # Find the latest model file
    RELOAD_PATH=$(find ./exps_cross_attn -name "*.pth" -path "*vfold${EVAL_FOLD}*heads${CROSS_ATTN_HEADS}*layers${CROSS_ATTN_LAYERS}*" -path "*/snapshots/*" | sort | tail -1)
    
    # Manual fallback paths for each fold if automatic detection fails
    if [ -z "$RELOAD_PATH" ] || [ ! -f "$RELOAD_PATH" ]; then
        echo "⚠️  Automatic model detection failed, trying manual paths..."
        
        case $EVAL_FOLD in
            0)
                RELOAD_PATH="E:\Suyash\iterative refinement\cowpro_supervised\exps_cross_attn\cross_attn_cowpro_0\CoWPro_cross_attn_train_CHAOST2_Supervised_lbgroup0_vfold0_heads8_layers2_f0_h8_l2\8\snapshots/50000.pth"
                ;;
            1)
                RELOAD_PATH="./exps_cross_attn/cross_attn_cowpro_0/cross_attn_train_CHAOST2_Supervised_lbgroup0_vfold1_heads8_layers2/1/snapshots/50000.pth"
                ;;
            2)
                RELOAD_PATH="./exps_cross_attn/cross_attn_cowpro_0/cross_attn_train_CHAOST2_Supervised_lbgroup0_vfold2_heads8_layers2/1/snapshots/50000.pth"
                ;;
            3)
                RELOAD_PATH="./exps_cross_attn/cross_attn_cowpro_0/cross_attn_train_CHAOST2_Supervised_lbgroup0_vfold3_heads8_layers2/1/snapshots/50000.pth"
                ;;
            4)
                RELOAD_PATH="./exps_cross_attn/cross_attn_cowpro_0/cross_attn_train_CHAOST2_Supervised_lbgroup0_vfold4_heads8_layers2/1/snapshots/50000.pth"
                ;;
            *)
                echo "❌ Unknown fold: $EVAL_FOLD"
                continue
                ;;
        esac
    fi
    
    # ===================================================================
    # CHECK IF MODEL EXISTS
    # ===================================================================
    
    if [ ! -f "$RELOAD_PATH" ]; then
        echo "❌ Model file not found for fold $EVAL_FOLD: $RELOAD_PATH"
        echo ""
        echo "🔍 Searching for available models for fold $EVAL_FOLD..."
        find ./exps_cross_attn -name "*.pth" -path "*vfold${EVAL_FOLD}*" -path "*/snapshots/*" | head -5
        echo ""
        echo "💡 Please check if training completed successfully for fold $EVAL_FOLD"
        echo "    or update the path manually in the script"
        echo ""
        continue
    fi
    
    echo "✅ Found model for fold $EVAL_FOLD: $RELOAD_PATH"
    
    # Extract model info for logging
    MODEL_SIZE=$(du -h "$RELOAD_PATH" | cut -f1)
    echo "📦 Model size: $MODEL_SIZE"
    
    # ===================================================================
    # RUN VALIDATION WITH CROSS-ATTENTION ANALYSIS
    # ===================================================================
    
    echo "🚀 Starting validation with cross-attention analysis..."
    
    python validation_supervised_cross_attn.py with \
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
    z_margin=0 \
    model.use_cross_attention=$USE_CROSS_ATTN \
    model.cross_attn_heads=$CROSS_ATTN_HEADS \
    model.cross_attn_layers=$CROSS_ATTN_LAYERS \
    model.cross_attn_dropout=$CROSS_ATTN_DROPOUT

    VALIDATION_EXIT_CODE=$?

    if [ $VALIDATION_EXIT_CODE -eq 0 ]; then
        echo "✅ Validation completed successfully for fold $EVAL_FOLD"
        echo "📁 Results saved in: $LOGDIR/$PREFIX"
        echo "📊 Standard prediction visualizations: $LOGDIR/$PREFIX/*/interm_preds/"
        echo "🎯 Cross-attention analysis: $LOGDIR/$PREFIX/*/attention_analysis/"
        echo "📈 Metrics logged and available in experiment directory"
    else
        echo "❌ Validation failed for fold $EVAL_FOLD (exit code: $VALIDATION_EXIT_CODE)"
        echo "📋 Check the logs for detailed error information"
    fi
    
    echo "========================================="
done

echo ""
echo "🎉 Validation completed for all folds!"
echo ""
echo "📁 All results saved in: $LOGDIR"
echo ""
echo "📊 Cross-Attention Analysis Available:"
echo "  - Standard prediction GIFs with overlays"
echo "  - Enhanced attention visualization GIFs"
echo "  - Detailed attention maps (numpy arrays)"
echo "  - Prototype assignment visualizations"
echo "  - Support-query interaction analysis"
echo ""
echo "📈 Performance Analysis:"
echo "Run the following to extract and analyze results:"
echo ""
echo "# Extract quantitative results"
echo "python extract_cross_attn_results.py --results_dir $LOGDIR --output cross_attn_validation_summary.csv"
echo ""
echo "# Analyze attention patterns"
echo "python analyze_attention_patterns.py --attention_dir $LOGDIR/*/attention_analysis/"
echo ""
echo "🔍 Model path verification:"
echo "To verify all trained models exist, run:"
echo "find ./exps_cross_attn -name '*.pth' -path '*/snapshots/*' | sort"
echo ""
echo "📊 Compare with baseline CoWPro:"
echo "Use the attention analysis to understand:"
echo "  - How cross-attention improves support-query alignment"
echo "  - Which regions receive higher attention weights"
echo "  - Prototype assignment patterns"
echo "  - Class-specific attention behaviors"
echo ""
echo "🎯 Expected improvements over baseline:"
echo "  - Better few-shot segmentation performance"
echo "  - More robust prototype matching"
echo "  - Improved generalization to unseen classes"
echo "  - Enhanced interpretability through attention maps"

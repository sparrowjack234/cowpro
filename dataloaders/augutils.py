"""
Fixed Configuration File with Short Experiment Names
Resolves Windows long path issues
"""
import os
import re
import glob
import itertools

import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

from platform import node
from datetime import datetime

# Configure Sacred to handle Windows paths better
sacred.SETTINGS['CONFIG']['READ_ONLY_CONFIG'] = False
sacred.SETTINGS.CAPTURE_MODE = 'no'

ex = Experiment('CoWPro')  # Shorter experiment name
ex.captured_out_filter = apply_backspaces_and_linefeeds

# More robust source file handling
source_folders = ['.', './dataloaders', './models', './util']
sources_to_save = []

for folder in source_folders:
    if os.path.exists(folder):
        # Use glob with recursive search and handle Windows paths
        pattern = os.path.join(folder, '*.py').replace('\\', '/')
        files = glob.glob(pattern)
        sources_to_save.extend(files)

# Only add source files that actually exist
for source_file in sources_to_save:
    if os.path.exists(source_file):
        try:
            ex.add_source_file(source_file)
        except Exception as e:
            print(f"Warning: Could not add source file {source_file}: {e}")

@ex.config
def cfg():
    """Default configurations with cross-attention support"""
    seed = 1234
    gpu_id = 0
    mode = 'train'
    num_workers = 4

    dataset = 'CHAOST2_Supervised'  # Updated for supervised training
    use_coco_init = True

    ### Cross-Attention Parameters ###
    use_cross_attention = True  # Enable/disable cross-attention
    cross_attn_heads = 8       # Number of attention heads
    cross_attn_layers = 2      # Number of cross-attention layers
    cross_attn_dropout = 0.1   # Dropout rate for cross-attention
    
    ### Training
    n_steps = 50000  # Reduced for supervised training
    batch_size = 1
    lr_milestones = [ii * 1000 for ii in range(1, n_steps // 1000)]
    lr_step_gamma = 0.95
    ignore_label = 255
    print_interval = 200  # More frequent printing for debugging
    save_snapshot_every = 10000
    max_iters_per_load = 1000
    scan_per_load = -1
    which_aug = 'aug_v3'  # More aggressive augmentation
    input_size = (256, 256)
    min_fg_data = 1  # Changed to int to avoid type warning
    label_sets = 0
    exclude_cls_list = [2, 3]  # Exclude kidneys for testing
    usealign = True
    use_wce = True
    use_tversky = True
    viz = 1
    
    ### Distance constraints for supervised training
    min_slice_distance = 4
    max_distance_ratio = 1/6
    
    ### Validation
    z_margin = 0
    eval_fold = 0
    support_idx = [0, 1, 2, 3, 4]  # Multiple support options
    val_wsize = 2
    n_sup_part = 3

    # Network
    modelname = 'dlfcn_res101'
    clsname = 'grid_proto'
    resume = False
    reload_model_path = ''
    proto_grid_size = 8
    feature_hw = [32, 32]

    # Loss weights
    tversky_params = {
        'tversky_alpha': 0.3,
        'tversky_beta': 0.7,
        'tversky_gamma': 1.0
    }

    lambda_loss = {
        'loss1': 0.0,  # Not used in supervised training
        'loss2': 1.0,  # Main segmentation loss
        'loss3': 0.0,  # Not used
        'loss4': 0.0,  # Not used
        'loss5': 0.0   # Not used
    }

    accum_iter = 1

    model = {
        'align': usealign,
        'use_coco_init': use_coco_init,
        'which_model': modelname,
        'cls_name': clsname,
        'proto_grid_size': proto_grid_size,
        'feature_hw': feature_hw,
        'reload_model_path': reload_model_path,
        # Cross-attention specific parameters
        'use_cross_attention': use_cross_attention,
        'cross_attn_heads': cross_attn_heads,
        'cross_attn_layers': cross_attn_layers,
        'cross_attn_dropout': cross_attn_dropout
    }

    task = {
        'n_ways': 1,
        'n_shots': 1,
        'n_queries': 1,
        'npart': n_sup_part
    }

    optim_type = 'sgd'
    optim = {
        'lr': 1e-3,
        'momentum': 0.9,
        'weight_decay': 0.0005,
    }

    exp_prefix = 'xattn'  # Much shorter prefix

    # MUCH SHORTER experiment string to avoid Windows path length limits
    exp_str = f'{exp_prefix}_f{eval_fold}_h{cross_attn_heads}_l{cross_attn_layers}'

    # Data paths - UPDATE THESE TO YOUR ACTUAL PATHS
    path = {
        'log_dir': './runs',  # Shorter path
        'SABS': {
            'data_dir': "E:\\Suyash\\cowpro\\data\\SABS\\sabs_CT_normalized"
        },
        'C0': {
            'data_dir': "E:\\Suyash\\cowpro\\data"
        },
        'CHAOST2': {
            'data_dir': "E:\\Suyash\\cowpro\\data\\CHAOST2\\chaos_MR_T2_normalized"
        },
        'FLARE22Train': {
            'data_dir': "E:\\Suyash\\cowpro\\data\\FLARE22Train\\flare_CT_normalized"
        },
        'SABS_Supervised': {
            'data_dir': "E:\\Suyash\\cowpro\\data\\SABS\\sabs_CT_normalized"
        },
        'C0_Superpix': {
            'data_dir': "E:\\Suyash\\cowpro\\data"
        },
        'CHAOST2_Supervised': {
            'data_dir': "E:\\Suyash\\cowpro\\data\\CHAOST2\\chaos_MR_T2_normalized"
        },
        'FLARE22Train_Supervised': {
            'data_dir': "E:\\Suyash\\cowpro\\data\\FLARE22Train\\flare_CT_normalized"
        }
    }

    DATASET_CONFIG = {
        'SABS': {
            'img_bname': f'E:\\Suyash\\cowpro\\data\\SABS\\sabs_CT_normalized/image_*.nii.gz',
            'out_dir': 'E:\\Suyash\\cowpro\\data\\SABS\\sabs_CT_normalized',
            'fg_thresh': 1e-4,
        },
        'CHAOST2': {
            'img_bname': f'E:\\Suyash\\cowpro\\data\\CHAOST2\\chaos_MR_T2_normalized/image_*.nii.gz',
            'out_dir': 'E:\\Suyash\\cowpro\\data\\CHAOST2\\chaos_MR_T2_normalized',
            'fg_thresh': 1e-4 + 50,
        },
        'FLARE22Train': {
            'img_bname': f'E:\\Suyash\\cowpro\\data\\FLARE22Train\\flare_CT_normalized/image_*.nii.gz',
            'out_dir': 'E:\\Suyash\\cowpro\\data\\FLARE22Train\\flare_CT_normalized',
            'fg_thresh': 1e-4                     
        },
    }


@ex.config_hook
def add_observer(config, command_name, logger):
    """A hook function to add observer with better path handling"""
    exp_name = f'{ex.path}_{config["exp_str"]}'
    
    # Ensure log directory exists and handle Windows paths
    log_dir = os.path.normpath(config['path']['log_dir'])
    os.makedirs(log_dir, exist_ok=True)
    
    exp_path = os.path.join(log_dir, exp_name)
    
    # Check if path is too long (Windows limit is ~260 characters)
    if len(exp_path) > 200:  # Leave some buffer
        # Create even shorter path
        import hashlib
        short_hash = hashlib.md5(exp_name.encode()).hexdigest()[:8]
        exp_name = f'{ex.path}_{short_hash}'
        exp_path = os.path.join(log_dir, exp_name)
        print(f"⚠️ Path too long, using shortened name: {exp_name}")
    
    try:
        observer = FileStorageObserver.create(exp_path)
        ex.observers.append(observer)
        print(f"✅ Experiment will be saved to: {exp_path}")
    except Exception as e:
        print(f"⚠️ Warning: Could not create file observer: {e}")
        print("Continuing without file observer...")
    
    return config

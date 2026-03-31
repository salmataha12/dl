def get_config():
    return {
        'MODEL': {
            'NAME': 'deit_tiny',        # Default: None
            'NUM_CLASSES': 5,           # Default: 5
            'DROP_PATH_RATE': 0.1,      # Default: 0.1
            'LABEL_SMOOTHING': 0.1,     # Default: 0.1

            # Vision Transformer specific parameters
            'IMG_SIZE': 224,            # Default: 224
            'PATCH_SIZE': 16,           # Default: 16
            'IN_CHANS': 3,              # Default: 3
            'EMBED_DIM': 192,           # Default: 192
            'DEPTH': 12,                # Default: 12
            'NUM_HEADS': 3,             # Default: 3    
            'MLP_RATIO': 4,             # Default: 4
            'QKV_BIAS': True,           # Default: True
        },
        'DATA': {
            'DATA_PATH': None,          # Default: 'food_subset'
            'BATCH_SIZE': None,         # Default: 64
            'NUM_WORKERS': None,        # Default: 2
            'PIN_MEMORY': None,         # Default: True
        },
        'TRAIN': {
            'START_EPOCH': None,        # Default: 0
            'EPOCHS': None,             # Default: 300
            'BASE_LR': None,            # Default: 5e-4
            'WEIGHT_DECAY': None,       # Default: 0.05
            'CLIP_GRAD': None,          # Default: 5.0
            'WARMUP_EPOCHS': None,      # Default: 5
            'WARMUP_LR': None,          # Default: 1e-6
            'MIN_LR': None,             # Default: 1e-5
            'OPT': None,                # Default: 'adamw'
            'SCHED': None,              # Default: 'cosine'
        },
        'OUTPUT': None,                 # Default: 'outputs'
    }

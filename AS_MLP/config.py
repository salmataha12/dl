def get_config():
    return {
        'MODEL': {
            'NAME': 'as_mlp_tiny',      # Default: None
            'TAG': 'MLP',               # set to one of those ['CNN', 'MLP', 'TRANSFORMER']
            'NUM_CLASSES': 5,           # Default: 5
            'DROP_PATH_RATE': 0.1,      # Default: 0.1
            'LABEL_SMOOTHING': 0.1,     # Default: 0.1

            # AS-MLP specific parameters
            'IMG_SIZE': 224,            # Default: 224
            'PATCH_SIZE': 4,            # Default: 4
            'IN_CHANS': 3,              # Default: 3
            'EMBED_DIM': 96,            # Default: 96
            'DEPTHS': [2, 2, 6, 2],     # Default: [2, 2, 6, 2]
            'SHIFT_SIZE': 5,            # Default: 5
            'MLP_RATIO': 4.,            # Default: 4.
            'AS_BIAS': True,            # Default: True
            'DROP_RATE': 0.,            # Default: 0.
            'PATCH_NORM': True,         # Default: True
        },
        'DATA': {
            'DATA_PATH': None,          # Default: 'food_subset'
            'BATCH_SIZE': None,         # Default: 64
            'NUM_WORKERS': None,        # Default: 2
            'PIN_MEMORY': None,         # Default: True
        },
        'TRAIN': {
            'START_EPOCH': None,        # Default: 0
            'EPOCHS': None,             # Default: 100
            'BASE_LR': None,            # Default: 5e-4
            'WEIGHT_DECAY': None,       # Default: 0.05
            'CLIP_GRAD': None,          # Default: 5.0
            'WARMUP_EPOCHS': 20,        # Default: 5
            'WARMUP_LR': None,          # Default: 1e-6
            'MIN_LR': None,             # Default: 1e-5
            'OPT': None,                # Default: 'adamw'
            'SCHED': None,              # Default: 'cosine'
        },
        'OUTPUT': None,                 # Default: 'outputs'
    }

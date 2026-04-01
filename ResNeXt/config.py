def get_config():
    return {
        'MODEL': {
            'NAME': 'resnext50_local',  # Default: None
            'TAG': 'CNN',
            'NUM_CLASSES': 5,           # Default: 5
            'DROP_PATH_RATE': 0.2,      # Increased from 0.1 for more regularization
            'LABEL_SMOOTHING': 0.1,     # Default: 0.1

            # ResNeXt specific parameters
            'NUM_BLOCKS': [3, 4, 6, 3], # Default: [3, 4, 6, 3]
            'CARDINALITY': 32,          # Default: 32
            'BOTTLENECK_WIDTH': 4,      # Default: 4
            'EXPANSION': 2,             # Default: 2
            'KERNEL_SIZE': 7,           # Default: 7
            'STRIDE': 2,                # Default: 2
            'PADDING': 3,               # Default: 3
        },
        'DATA': {
            'DATA_PATH': None,          # Default: 'food_subset'
            'BATCH_SIZE': None,         # Default: 64
            'NUM_WORKERS': None,        # Default: 2
            'PIN_MEMORY': None,         # Default: True
        },
        'TRAIN': {
            'START_EPOCH': None,        # Default: 0
            'EPOCHS': 100,              # Unified with other models for comparison
            'BASE_LR': 5e-4,            # Reduced from 0.1 for better generalization
            'WEIGHT_DECAY': 1e-3,       # Increased from 5e-4 for more regularization
            'CLIP_GRAD': 1.0,           # Added gradient clipping
            'WARMUP_EPOCHS': 5,         # Added warmup from 0
            'WARMUP_LR': 1e-6,          # Warmup learning rate
            'MIN_LR': 1e-6,             # Minimum learning rate
            'OPT': 'adamw',             # Changed from sgd to adamw
            'SCHED': 'cosine',          # Changed from step to cosine
        },
        'OUTPUT': None,                 # Default: 'outputs'
    }

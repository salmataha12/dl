def get_config():
    return {
        'MODEL': {
            'NAME': 'resnext50_local',  # Default: None
            'TAG': 'CNN',               # set to one of those ['CNN', 'MLP', 'TRANSFORMER']
            'NUM_CLASSES': 5,           # Default: 5
            'DROP_PATH_RATE': 0.15,     # Default: 0.1
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
            'EPOCHS': 100,              # Default: 100
            'BASE_LR': 5e-4,            # Default: 5e-4
            'WEIGHT_DECAY': 5e-4,       # Default: 0.05
            'CLIP_GRAD': 1.0,           # Default: 5.0
            'WARMUP_EPOCHS': 5,         # Default: 5
            'WARMUP_LR': 1e-6,          # Default: 1e-6
            'MIN_LR': 1e-6,             # Default: 1e-5
            'OPT': 'adamw',             # Default: 'adamw'
            'SCHED': 'cosine',          # Default: 'cosine'
        },
        'OUTPUT': None,                 # Default: 'outputs'
    }

def get_config():
    return {
        'MODEL': {
            'NAME': 'gmlp_tiny',
            'TAG': 'MLP',
            'NUM_CLASSES': 5,
            'DROP_PATH_RATE': 0.1,
            'LABEL_SMOOTHING': 0.1,
            
            # gMLP specific parameters
            'IMG_SIZE': 224,
            'PATCH_SIZE': 16,
            'IN_CHANS': 3,
            'EMBED_DIM': 256,
            'DEPTH': 12,
            'MLP_RATIO': 4.,
        },
        'DATA': {
            'DATA_PATH': None,
            'BATCH_SIZE': 64,
            'NUM_WORKERS': None,
            'PIN_MEMORY': None,
        },
        'TRAIN': {
            'START_EPOCH': None,
            'EPOCHS': 50,
            'BASE_LR': 1e-3,            # MLP-based models benefit from higher LR
            'WEIGHT_DECAY': 1e-4,
            'CLIP_GRAD': 1.0,
            'WARMUP_EPOCHS': 20,        # Longer warmup for MLP
            'WARMUP_LR': 1e-6,
            'MIN_LR': 1e-5,
            'OPT': 'adamw',
            'SCHED': 'cosine',
        },
        'OUTPUT': None,
    }
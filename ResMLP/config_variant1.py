def get_config():
    return {
        'MODEL': {
            'NAME': 'resmlp_s12_v1',
            'TAG': 'MLP',
            'NUM_CLASSES': 5,
            'EMBED_DIM': 384,
            'DEPTH': 12,
            'MLP_RATIO': 4.0,
            'DROP_PATH_RATE': 0.05,       # reduced to ease overfitting
            'LABEL_SMOOTHING': 0.1,
        },
        'DATA': {
            'DATA_PATH': None,
            'BATCH_SIZE': 32,
            'NUM_WORKERS': 2,
            'PIN_MEMORY': True,
        },
        'TRAIN': {
            'START_EPOCH': 0,
            'EPOCHS': 100,
            'BASE_LR': 2e-4,              # slightly lower for stability
            'WEIGHT_DECAY': 0.05,
            'CLIP_GRAD': 1.0,
            'WARMUP_EPOCHS': 15,          # longer warmup to reduce gap
            'WARMUP_LR': 1e-6,
            'MIN_LR': 1e-5,
            'OPT': 'adamw',
            'SCHED': 'cosine',
        },
        'OUTPUT': None,
    }
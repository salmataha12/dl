def get_config():
    return {
        'MODEL': {
            'NAME': 'resnet18_v1',
            'TAG': 'CNN',
            'NUM_CLASSES': 5,
            'DROP_PATH_RATE': 0.05,  # Reduced from 0.1
            'LABEL_SMOOTHING': 0.1,
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
            'BASE_LR': 5e-4,
            'WEIGHT_DECAY': 5e-4,
            'CLIP_GRAD': 1.0,
            'WARMUP_EPOCHS': 5,
            'WARMUP_LR': 1e-6,
            'MIN_LR': 1e-6,
            'OPT': 'adamw',
            'SCHED': 'cosine',
        },
        'OUTPUT': None,
    }
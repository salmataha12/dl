def get_config():
    return {
        'MODEL': {
            'NAME': 'resnet18',
            'TAG': 'CNN',
            'NUM_CLASSES': 5,
            'DROP_PATH_RATE': 0.1,
            'LABEL_SMOOTHING': 0.1,
            
            # ResNet-18 specific parameters (standard defaults)
        },
        'DATA': {
            'DATA_PATH': None,          # Use default from base config
            'BATCH_SIZE': 64,           # Good batch size for ResNet
            'NUM_WORKERS': None,
            'PIN_MEMORY': None,
        },
        'TRAIN': {
            'START_EPOCH': None,
            'EPOCHS': 100,
            'BASE_LR': 5e-4,            # Learning rate for fine-tuning
            'WEIGHT_DECAY': 5e-4,       # L2 regularization
            'CLIP_GRAD': 1.0,           # Gradient clipping
            'WARMUP_EPOCHS': 5,         # Warmup period
            'WARMUP_LR': 1e-6,
            'MIN_LR': 1e-6,
            'OPT': 'adamw',             # Optimizer
            'SCHED': 'cosine',          # Learning rate scheduler
        },
        'OUTPUT': None,
    }
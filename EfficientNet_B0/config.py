def get_config(variant=None):    
    if variant == 'efficientnet_b0' or variant is None:
        return {
            'MODEL': {
                'NAME': 'efficientnet_b0',
                'TAG': 'CNN',
                'NUM_CLASSES': 5,
                'DROP_PATH_RATE': 0.1,
                'LABEL_SMOOTHING': 0.1,
                'PRETRAINED': True,
                'FREEZE_BACKBONE': True,
            },
            'DATA': {
                'DATA_PATH': None,
                'BATCH_SIZE': 64,
                'NUM_WORKERS': 4,
                'PIN_MEMORY': True,
            },
            'TRAIN': {
                'START_EPOCH': None,
                'EPOCHS': 50,
                'BASE_LR': 1e-4,
                'WEIGHT_DECAY': 0.01,  
                'CLIP_GRAD': 5.0,
                'WARMUP_EPOCHS': 5,
                'WARMUP_LR': 1e-6,
                'MIN_LR': 1e-5,
                'OPT': 'adamw',
                'SCHED': 'cosine', 
            },
            'OUTPUT': None,
        }
    
    elif variant == 'efficientnet_b0_v2':
        return {
            'MODEL': {
                'NAME': 'efficientnet_b0_v2',
                'TAG': 'CNN',
                'NUM_CLASSES': 5,
                'DROP_PATH_RATE': 0.1,
                'LABEL_SMOOTHING': 0.1,
                
                'PRETRAINED': True,
                'FREEZE_BACKBONE': True,
            },
            'DATA': {
                'DATA_PATH': None,
                'BATCH_SIZE': 64,
                'NUM_WORKERS': 4,
                'PIN_MEMORY': True,
            },
            'TRAIN': {
                'START_EPOCH': None,
                'EPOCHS': 50,
                'BASE_LR': 1e-4,
                'WEIGHT_DECAY': 1e-4,  # reduced from 0.01
                'CLIP_GRAD': 5.0,
                'WARMUP_EPOCHS': 5,
                'WARMUP_LR': 1e-6,
                'MIN_LR': 1e-5,
                'OPT': 'adamw',
                'SCHED': 'cosine',
            },
            'OUTPUT': None,
        }
    else:
        raise ValueError(f"Unknown variant for EfficientNet-B0: {variant}")

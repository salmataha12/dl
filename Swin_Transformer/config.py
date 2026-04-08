def get_config(variant=None):   
    if variant == 'swin_transformer' or variant is None:
        return {
            'MODEL': {
                'NAME': 'swin_transformer',
                'TAG': 'Transformer',
                'NUM_CLASSES': 5,
                'DROP_PATH_RATE': 0.1,
                'LABEL_SMOOTHING': 0.1,
 
                'IMG_SIZE': 224,
                'PATCH_SIZE': 4,
                'IN_CHANS': 3,
                'EMBED_DIM': 96,
                'DEPTHS': [2, 2, 6, 2], 
                'NUM_HEADS': [3, 6, 12, 24],  
                'WINDOW_SIZE': 7,
                'MLP_RATIO': 4.0,
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
                'BASE_LR': 3e-4, 
                'WEIGHT_DECAY': 0.003,  
                'CLIP_GRAD': 5.0,
                'WARMUP_EPOCHS': 5,
                'WARMUP_LR': 1e-6,
                'MIN_LR': 1e-4,
                'OPT': 'adamw',
                'SCHED': 'cosine', 
            },
            'OUTPUT': None,
        }
    
    elif variant == 'swin_transformer_v2':
        return {
            'MODEL': {
                'NAME': 'swin_transformer_v2',
                'TAG': 'Transformer',
                'NUM_CLASSES': 5,
                'DROP_PATH_RATE': 0.1,
                'LABEL_SMOOTHING': 0.1,
                
                'IMG_SIZE': 224,
                'PATCH_SIZE': 4,
                'IN_CHANS': 3,
                'EMBED_DIM': 96,
                'DEPTHS': [2, 2, 6, 2],
                'NUM_HEADS': [3, 6, 12, 24],
                'WINDOW_SIZE': 7,
                'MLP_RATIO': 4.0,
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
                'BASE_LR': 5e-4,  # increased from 3e-4
                'WEIGHT_DECAY': 0.003,
                'CLIP_GRAD': 5.0,
                'WARMUP_EPOCHS': 5,
                'WARMUP_LR': 1e-6,
                'MIN_LR': 1e-4,
                'OPT': 'adamw',
                'SCHED': 'cosine',
            },
            'OUTPUT': None,
        }
    else:
        raise ValueError(f"Unknown variant for Swin Transformer: {variant}")
def get_config():
    return {
        'MODEL': {
            'NAME': 'vit_base',
            'TAG': 'TRANSFORMER',
            'NUM_CLASSES': 5,
            'DROP_PATH_RATE': 0.1,
            'LABEL_SMOOTHING': 0.1,
            
            # ViT specific parameters
            'IMG_SIZE': 224,
            'PATCH_SIZE': 16,
            'IN_CHANS': 3,
            'EMBED_DIM': 768,
            'DEPTH': 12,
            'NUM_HEADS': 12,
            'MLP_RATIO': 4.,
        },
        'DATA': {
            'DATA_PATH': None,
            'BATCH_SIZE': 32,           # Smaller batch for transformer (memory intensive)
            'NUM_WORKERS': None,
            'PIN_MEMORY': None,
        },
        'TRAIN': {
            'START_EPOCH': None,
            'EPOCHS': 50,
            'BASE_LR': 1e-3,            # ViT needs careful learning rate tuning
            'WEIGHT_DECAY': 0.05,
            'CLIP_GRAD': 1.0,
            'WARMUP_EPOCHS': 20,        # Long warmup for transformers
            'WARMUP_LR': 1e-6,
            'MIN_LR': 1e-5,
            'OPT': 'adamw',
            'SCHED': 'cosine',
        },
        'OUTPUT': None,
    }
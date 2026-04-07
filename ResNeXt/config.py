def get_config(variant=None): # resnext50_32x4d and resnext101_32x8d
    if variant == 'resnext101_32x8d':
        return {
            'MODEL': {
                'NAME': 'resnext101_local',
                'TAG': 'CNN',
                'NUM_CLASSES': 5,
                'DROP_PATH_RATE': 0.15,
                'LABEL_SMOOTHING': 0.1,
                'NUM_BLOCKS': [3, 4, 23, 3],
                'CARDINALITY': 32,
                'BOTTLENECK_WIDTH': 8,
                'EXPANSION': 2,
                'KERNEL_SIZE': 7,
                'STRIDE': 2,
                'PADDING': 3
            },
            'DATA': {
                'DATA_PATH': None,
                'BATCH_SIZE': None,
                'NUM_WORKERS': None,
                'PIN_MEMORY': None
            },
            'TRAIN': {
                'START_EPOCH': None,
                'EPOCHS': 100,
                'BASE_LR': 5e-4,
                'WEIGHT_DECAY': 5e-4,
                'CLIP_GRAD': 1.0,
                'WARMUP_EPOCHS': 5,
                'WARMUP_LR': 1e-6,
                'MIN_LR': 1e-6,
                'OPT': 'adamw',
                'SCHED': 'cosine'
            },
            'OUTPUT': None
        }
    elif variant == 'resnext50_32x4d':
        return {
            'MODEL': {
                'NAME': 'resnext50_local',
                'TAG': 'CNN',
                'NUM_CLASSES': 5,
                'DROP_PATH_RATE': 0.15,
                'LABEL_SMOOTHING': 0.1,
                'NUM_BLOCKS': [3, 4, 6, 3],
                'CARDINALITY': 32,
                'BOTTLENECK_WIDTH': 4,
                'EXPANSION': 2,
                'KERNEL_SIZE': 7,
                'STRIDE': 2,
                'PADDING': 3
            },
            'DATA': {
                'DATA_PATH': None,
                'BATCH_SIZE': None,
                'NUM_WORKERS': None,
                'PIN_MEMORY': None
            },
            'TRAIN': {
                'START_EPOCH': None,
                'EPOCHS': 100,
                'BASE_LR': 5e-4,
                'WEIGHT_DECAY': 5e-4,
                'CLIP_GRAD': 1.0,
                'WARMUP_EPOCHS': 5,
                'WARMUP_LR': 1e-6,
                'MIN_LR': 1e-6,
                'OPT': 'adamw',
                'SCHED': 'cosine'
            },
            'OUTPUT': None
        }

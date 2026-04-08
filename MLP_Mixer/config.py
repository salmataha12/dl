def get_config(variant=None):
    if variant == 'mlp_mixer' or variant is None:
        return {
            'MODEL': {
                'NAME': 'mlp_mixer',
                'TAG': 'MLP',
                'NUM_CLASSES': 5,
                'LABEL_SMOOTHING': 0.0,
                'PRETRAINED': True,
                'FREEZE_BACKBONE': True,
            },
            'DATA': {
                'DATA_PATH': None,
                'BATCH_SIZE': 64,
                'NUM_WORKERS': None,
                'PIN_MEMORY': None,
            },
            'TRAIN': {
                'START_EPOCH': None,
                'EPOCHS': 20,
                'BASE_LR': 1e-2,
                'WEIGHT_DECAY': 1e-4,
                'CLIP_GRAD': 0.0,
                'WARMUP_EPOCHS': 0,
                'WARMUP_LR': 1e-6,
                'MIN_LR': 1e-5,
                'OPT': 'adamw',
                'SCHED': 'step',
                'DECAY_EPOCHS': 5,
                'DECAY_RATE': 0.5,
            },
            'OUTPUT': None,
            'AMP_ENABLE': False,
        }

    elif variant == 'mlp_mixer_v2':
        return {
            'MODEL': {
                'NAME': 'mlp_mixer_v2',
                'TAG': 'MLP',
                'NUM_CLASSES': 5,
                'LABEL_SMOOTHING': 0.0,
            },
            'DATA': {
                'DATA_PATH': None,
                'BATCH_SIZE': 64,
                'NUM_WORKERS':4,
                'PIN_MEMORY': True,
            },
            'TRAIN': {
                'START_EPOCH': None,
                'EPOCHS': 20,
                'BASE_LR': 5e-3,           # reduced from 1e-2
                'WEIGHT_DECAY': 1e-4,
                'CLIP_GRAD': 0.0,
                'WARMUP_EPOCHS': 0,
                'WARMUP_LR': 1e-6,
                'MIN_LR': 1e-5,
                'OPT': 'adamw',
                'SCHED': 'step',
                'DECAY_EPOCHS': 5,
                'DECAY_RATE': 0.5,
            },
            'OUTPUT': None,
            'AMP_ENABLE': False,
        }
    else:
        raise ValueError(f"Unknown variant for MLP-Mixer: {variant}")
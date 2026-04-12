def get_config():
    return {
        'MODEL': {
            'NAME': 'regnety_8gf_v1',
            'TAG': 'CNN',
            'NUM_CLASSES': 5,
            'DROP_PATH_RATE': 0.15,       # slightly weaker regularization
            'LABEL_SMOOTHING': 0.1,      
        },

        'DATA': {
            'DATA_PATH': None,
            'BATCH_SIZE': 32,
            'NUM_WORKERS': None,
            'PIN_MEMORY': None,
        },

        'TRAIN': {
            'START_EPOCH': 0,
            'EPOCHS': 100,
            'BASE_LR': 2.5e-4,            # smoother convergence
            'WEIGHT_DECAY': 8e-5,         # weaker regularization
            'CLIP_GRAD': 1.0,
            'WARMUP_EPOCHS': 3,           # shorter warmup
            'WARMUP_LR': 1e-6,
            'MIN_LR': 1e-6,

            'OPT': 'adamw',
            'SCHED': 'cosine',
        },

        'OUTPUT': None,
    }

# PVT/config_variation.py

def get_config():
    return {
        'MODEL': {
            'NAME': 'pvt_v2_b0_regularized',   
            'TAG': 'TRANSFORMER',
            'NUM_CLASSES': 5,
            'DROP_PATH_RATE': 0.2,             # ↑ regularization
            'LABEL_SMOOTHING': 0.1,
        },
        'DATA': {
            'DATA_PATH': None,
            'BATCH_SIZE': None,
            'NUM_WORKERS': None,
            'PIN_MEMORY': None,
        },
        'TRAIN': {
            'START_EPOCH': 0,
            'EPOCHS': 100,
            'BASE_LR': 3e-4,                  # ↓ lower LR
            'WEIGHT_DECAY': 5e-4,             # ↑ stronger regularization
            'CLIP_GRAD': 5.0,
            'WARMUP_EPOCHS': 3,               # ↓ shorter warmup
            'WARMUP_LR': 1e-6,
            'MIN_LR': 1e-5,
            'OPT': 'adamw',
            'SCHED': 'cosine',
        },
        'OUTPUT': None,
    }
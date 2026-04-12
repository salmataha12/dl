def get_config():
    return {
        'MODEL': {
            'NAME': 'cvt_13_v1',
            'TAG': 'Transformer',
            'NUM_CLASSES': 5,
            'DROP_PATH_RATE': 0.2,       # stronger stochastic depth to reduce overfitting
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
            'BASE_LR': 2e-4,             # lower LR for smoother convergence
            'WEIGHT_DECAY': 5e-5,        # weaker regularization to balance stronger drop path
            'CLIP_GRAD': 1.0,
            'WARMUP_EPOCHS': 5,          # longer warmup to stabilize lower LR
            'WARMUP_LR': 1e-5,
            'MIN_LR': 1e-6,
            'OPT': 'adamw',
            'SCHED': 'cosine',
        },

        'OUTPUT': None,
    }

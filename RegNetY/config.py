def get_config():
    return {
        'MODEL': {
            'NAME': 'regnety_8gf',
            'TAG': 'CNN',
            'NUM_CLASSES': 5,
            'DROP_PATH_RATE': 0.2,      
            'LABEL_SMOOTHING': 0.1,
        },

        'DATA': {
            'DATA_PATH': None,          
            'BATCH_SIZE': 32,           
            'NUM_WORKERS': None,
            'PIN_MEMORY': None,
        },

        'TRAIN': {
            'START_EPOCH': None,
            'EPOCHS': 100,
            'BASE_LR': 3e-4,            
            'WEIGHT_DECAY': 1e-4,       

            'CLIP_GRAD': 1.0,           
            'WARMUP_EPOCHS': 5,
            'WARMUP_LR': 1e-6,
            'MIN_LR': 1e-6,
            'OPT': 'adamw',
            'SCHED': 'cosine',
        },

        'OUTPUT': None,
    }
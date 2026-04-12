def get_config():
    return {
        'MODEL': {
            'NAME': 'cvt_13',
            'TAG': 'Transformer',
            'NUM_CLASSES': 5,
            'DROP_PATH_RATE': 0.1,      
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
            'BASE_LR': 3e-4,               
            'WEIGHT_DECAY': 1e-4,          
            'CLIP_GRAD': 1.0,              
            'WARMUP_EPOCHS': 3,            
            'WARMUP_LR': 1e-5,
            'MIN_LR': 1e-6,                
            'OPT': 'adamw',
            'SCHED': 'cosine',             
        },

        'OUTPUT': None,            
    }
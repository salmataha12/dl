import os

# Base Configuration with all possible hyperparameters
BASE_CONFIG = {
    'DATA': {
        'DATA_PATH': 'food_subset',
        'BATCH_SIZE': 64,
        'NUM_WORKERS': 2,
        'PIN_MEMORY': True,
    },
    'MODEL': {
        'NAME': None, # Must be specified by model config
        'TAG': None,  # Architecture type (CNN, MLP, Transformer)
        'NUM_CLASSES': 5,
        'DROP_PATH_RATE': 0.1,
        'LABEL_SMOOTHING': 0.1,
    },
    'TRAIN': {
        'START_EPOCH': 0,
        'EPOCHS': 100,
        'BASE_LR': 5e-4,
        'WEIGHT_DECAY': 0.05,
        'CLIP_GRAD': 5.0,
        'WARMUP_EPOCHS': 5,
        'WARMUP_LR': 1e-6,
        'MIN_LR': 1e-5,
        'OPT': 'adamw',
        'SCHED': 'cosine',
        'MOMENTUM': 0.9,
        'EPS': 1e-8,
        'BETAS': (0.9, 0.999),
        'COOLDOWN_EPOCHS': 0,
        'DECAY_EPOCHS': 30,
        'DECAY_RATE': 0.1,
    },
    'OUTPUT': 'outputs',
    'AMP_ENABLE': True,
}

class Config:
    def __init__(self, model_config):
        # Merge model_config into BASE_CONFIG
        for section, params in model_config.items():
            if isinstance(params, dict) and section in BASE_CONFIG and isinstance(BASE_CONFIG[section], dict):
                for k, v in params.items():
                    if v is not None:
                        BASE_CONFIG[section][k] = v
            elif params is not None:
                BASE_CONFIG[section] = params
        
        # Set attributes from merged config
        self.__dict__.update(BASE_CONFIG)
        
        # Convert dicts to simple objects for dot notation
        self.DATA = type('', (), self.DATA)()
        self.MODEL = type('', (), self.MODEL)()
        self.TRAIN = type('', (), self.TRAIN)()
        
        # Ensure output dir exists
        os.makedirs(self.OUTPUT, exist_ok=True)

    def defrost(self): pass
    def freeze(self): pass

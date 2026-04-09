# ConvMLP/config_variation.py 
"""
ConvMLP-S Variation: Lower Learning Rate + Longer Warmup
VARIATION: Optimize convergence speed and earlier peak accuracy

Original Issues Fixed:
- Original took 97 epochs to peak (too slow)
- Original had high train loss (0.61+)
- Original needed 69 minutes training

This variant aims to:
- Achieve better accuracy EARLIER (epoch 30-50 range)
- Reduce training time
- Lower train loss through better learning rate tuning
"""

def get_config():
    return {
        'MODEL': {
            'NAME': 'convmlp_s_v2',
            'TAG': 'MLP',
            'NUM_CLASSES': 5,
            'DIM': 96,  
            'DROP_PATH_RATE': 0.1,           
            'LABEL_SMOOTHING': 0.1,         
        },
        'DATA': {
            'DATA_PATH': None,
            'BATCH_SIZE': None,
            'NUM_WORKERS': None,
            'PIN_MEMORY': None,
        },
        'TRAIN': {
            'START_EPOCH': None,
            'EPOCHS': 100,
            'BASE_LR': 5e-4,                 # ← REDUCED from 1e-3 (2x lower)
            'WEIGHT_DECAY': 1e-4,            
            'CLIP_GRAD': 5.0,                
            'WARMUP_EPOCHS': 20,             # ← INCREASED from 10 (2x longer warmup)
            'WARMUP_LR': 1e-6,               
            'MIN_LR': 1e-5,                  
            'OPT': 'adamw',               
            'SCHED': 'cosine',              
        },
        'OUTPUT': None,
    }
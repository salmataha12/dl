# ConvMLP/config_variation.py
"""
ConvMLP-S Variation: Wider Channels (dim: 64 → 96)
VARIATION: Increased model capacity for better feature learning

Original Architecture:
- dim = 64 (channels)
- Total stages have 64 → 128 → 256 → 512 channels

Variant Architecture:
- dim = 96 (channels) 
- Total stages have 96 → 192 → 384 → 768 channels
- ~1.5x more parameters

Rationale:
The original ConvMLP-S plateaued early (peak at epoch 97).
Increasing channel dimension gives model more capacity to learn
complex food features, potentially improving accuracy.

"""

def get_config():
    return {
        'MODEL': {
            'NAME': 'convmlp_s_v1',
            'TAG': 'MLP',
            'NUM_CLASSES': 5,
            'DIM': 96,                       #  ONLY THIS IS DIFFERENT (64 → 96)
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
            'BASE_LR': 1e-3,                 
            'WEIGHT_DECAY': 1e-4,           
            'CLIP_GRAD': 5.0,                
            'WARMUP_EPOCHS': 10,             
            'WARMUP_LR': 1e-6,               
            'MIN_LR': 1e-5,               
            'OPT': 'adamw',                 
            'SCHED': 'cosine',              
        },
        'OUTPUT': None,
    }
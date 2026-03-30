import os
import time
import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
import argparse

# Local imports
from models import build_model
from data import build_loader
from utils import load_checkpoint, save_checkpoint, get_grad_norm, reduce_tensor, create_logger

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
        'NUM_CLASSES': 5,
        'RESUME': None,
        'DROP_PATH_RATE': 0.1,
        'LABEL_SMOOTHING': 0.1,
    },
    'TRAIN': {
        'START_EPOCH': 0,
        'EPOCHS': 300,
        'BASE_LR': 5e-4,
        'WEIGHT_DECAY': 0.05,
        'CLIP_GRAD': 5.0,
        'WARMUP_EPOCHS': 5,
        'WARMUP_LR': 1e-6,
        'MIN_LR': 1e-5,
        'OPT': 'adamw',
        'SCHED': 'cosine',
    },
    'OUTPUT': 'outputs',
    'LOCAL_RANK': 0,
    'EVAL_MODE': False,
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
        
        # Specific output dir
        self.OUTPUT = os.path.join(self.OUTPUT, self.MODEL.NAME)
        os.makedirs(self.OUTPUT, exist_ok=True)

    def defrost(self): pass
    def freeze(self): pass

def main(config, logger):
    dataset_train, dataset_val, data_loader_train, data_loader_val = build_loader(config)
    
    logger.info(f"Creating model: {config.MODEL.NAME}")
    model = build_model(config)
    model.cuda()
    
    optimizer = create_optimizer(argparse_namespace(opt=config.TRAIN.OPT, lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY), model)
    scaler = torch.cuda.amp.GradScaler(enabled=config.AMP_ENABLE)
    
    if dist.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    
    # Scheduler setup
    lr_scheduler, _ = create_scheduler(argparse_namespace(
        sched=config.TRAIN.SCHED, 
        epochs=config.TRAIN.EPOCHS, 
        warmup_epochs=config.TRAIN.WARMUP_EPOCHS, 
        warmup_lr=config.TRAIN.WARMUP_LR, 
        min_lr=config.TRAIN.MIN_LR, 
        cooldown_epochs=0
    ), optimizer)

    max_accuracy = 0.0
    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model, optimizer, lr_scheduler, logger)

    if config.EVAL_MODE:
        acc1, acc5, loss = validate(config, data_loader_val, model, logger)
        logger.info(f"Accuracy on val set: {acc1:.1f}%")
        return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        if dist.is_initialized():
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, data_loader_train, optimizer, epoch, lr_scheduler, scaler, logger)
        
        if config.LOCAL_RANK == 0 and (epoch % 10 == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger)

        acc1, acc5, loss = validate(config, data_loader_val, model, logger)
        logger.info(f"Accuracy on val set: {acc1:.1f}%")
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

def train_one_epoch(config, model, data_loader, optimizer, epoch, lr_scheduler, scaler, logger):
    model.train()
    optimizer.zero_grad()
    
    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    
    start = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        samples, targets = samples.cuda(non_blocking=True), targets.cuda(non_blocking=True)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = model(samples)
            loss = torch.nn.CrossEntropyLoss()(outputs, targets)

        scaler.scale(loss).backward()
        if config.TRAIN.CLIP_GRAD:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        lr_scheduler.step_update(epoch * num_steps + idx)
        
        loss_meter.update(loss.item(), targets.size(0))
        batch_time.update(time.time() - start)
        start = time.time()

        if idx % 20 == 0 and config.LOCAL_RANK == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(f'Epoch: [{epoch}][{idx}/{num_steps}] lr {lr:.6f} loss {loss_meter.avg:.4f}')

@torch.no_grad()
def validate(config, data_loader, model, logger=None):
    model.eval()
    acc1_meter, loss_meter = AverageMeter(), AverageMeter()

    for images, target in data_loader:
        images, target = images.cuda(non_blocking=True), target.cuda(non_blocking=True)
        output = model(images)
        loss = torch.nn.CrossEntropyLoss()(output, target)
        acc1, _ = accuracy(output, target, topk=(1, 5))

        if dist.is_initialized():
            acc1, loss = reduce_tensor(acc1), reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))

    return acc1_meter.avg, 0, loss_meter.avg

class argparse_namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def parse_option():
    parser = argparse.ArgumentParser('Unified Training Script', add_help=False)
    parser.add_argument('--model_to_run', type=str, required=True, help='Model name to run (e.g., as_mlp_tiny, deit_tiny, resnext50_local)')
    # Add back any other general arguments that were in the original parse_option if needed, or stick to this minimum.
    # For now, let's keep it simple and just add the model_to_run.
    
    # Placeholder for other common args if needed, or remove them entirely
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for DistributedDataParallel')
    parser.add_argument('--eval_mode', action='store_true', help='Perform evaluation only')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint for resuming training/evaluation')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_option()
    MODEL_TO_RUN = args.model_to_run
    
    # Configure config.LOCAL_RANK and config.EVAL_MODE based on args
    LOCAL_RANK = args.local_rank
    EVAL_MODE = args.eval_mode
    RESUME = args.resume

    # Now load model-specific config
    if MODEL_TO_RUN == 'as_mlp_tiny':
        from AS_MLP.config import get_config
    elif MODEL_TO_RUN == 'deit_tiny':
        from DeiT.config import get_config
    elif MODEL_TO_RUN == 'resnext50_local':
        from ResNeXt.config import get_config
    else:
        raise ValueError(f"Unknown model: {MODEL_TO_RUN}")
    
    model_config_data = get_config()
    
    # Update config defaults from command line arguments
    model_config_data['MODEL']['RESUME'] = RESUME
    model_config_data['LOCAL_RANK'] = LOCAL_RANK
    model_config_data['EVAL_MODE'] = EVAL_MODE

    config = Config(model_config_data)
    
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl', init_method='env://')
    
    torch.cuda.set_device(config.LOCAL_RANK)
    seed = 42 + config.LOCAL_RANK
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    logger = create_logger(output_dir=config.OUTPUT, dist_rank=config.LOCAL_RANK, name=config.MODEL.NAME)
    main(config, logger)

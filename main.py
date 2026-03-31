import os
import time
import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import csv
from timm.utils import accuracy, AverageMeter
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler

# Local imports
from models import build_model, get_model_config
from data_prep import build_loader
from utils import create_logger, argparse_namespace, parse_option
from config import Config

def main(config, logger):
    dataset_train, dataset_val, data_loader_train, data_loader_val = build_loader(config)
    
    logger.info(f"Creating model: {config.MODEL.NAME}")
    model = build_model(config)
    model.cuda()
    
    optimizer = create_optimizer(argparse_namespace(
        opt=config.TRAIN.OPT, 
        lr=config.TRAIN.BASE_LR, 
        weight_decay=config.TRAIN.WEIGHT_DECAY,
        momentum=config.TRAIN.MOMENTUM,
        opt_eps=config.TRAIN.EPS,
        opt_betas=config.TRAIN.BETAS
    ), model)
    scaler = torch.cuda.amp.GradScaler(enabled=config.AMP_ENABLE)
    
    # Scheduler setup
    lr_scheduler, _ = create_scheduler(argparse_namespace(
        sched=config.TRAIN.SCHED, 
        epochs=config.TRAIN.EPOCHS, 
        warmup_epochs=config.TRAIN.WARMUP_EPOCHS, 
        warmup_lr=config.TRAIN.WARMUP_LR, 
        min_lr=config.TRAIN.MIN_LR, 
        cooldown_epochs=config.TRAIN.COOLDOWN_EPOCHS,
        decay_epochs=config.TRAIN.DECAY_EPOCHS,
        decay_rate=config.TRAIN.DECAY_RATE
    ), optimizer)

    # Setup metrics logging to file
    log_file = os.path.join(config.OUTPUT, 'metrics.csv')
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])

    max_accuracy = 0.0
    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        train_loss, train_acc = train_one_epoch(config, model, data_loader_train, optimizer, epoch, lr_scheduler, scaler, logger)
        
        val_acc, val_loss = validate(config, data_loader_val, model, logger)
        
        logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save metrics
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])

        max_accuracy = max_accuracy if max_accuracy > val_acc else val_acc
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

def train_one_epoch(config, model, data_loader, optimizer, epoch, lr_scheduler, scaler, logger):
    model.train()
    optimizer.zero_grad()
    
    num_steps = len(data_loader)
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
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
        
        acc1, _ = accuracy(outputs, targets, topk=(1, 5))
        loss_meter.update(loss.item(), targets.size(0))
        acc_meter.update(acc1.item(), targets.size(0))

        if idx % 20 == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(f'Epoch: [{epoch}][{idx}/{num_steps}] lr {lr:.6f} loss {loss_meter.avg:.4f} acc {acc_meter.avg:.2f}%')

    return loss_meter.avg, acc_meter.avg

@torch.no_grad()
def validate(config, data_loader, model, logger=None):
    model.eval()
    acc1_meter, loss_meter = AverageMeter(), AverageMeter()

    for images, target in data_loader:
        images, target = images.cuda(non_blocking=True), target.cuda(non_blocking=True)
        output = model(images)
        loss = torch.nn.CrossEntropyLoss()(output, target)
        acc1, _ = accuracy(output, target, topk=(1, 5))

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))

    return acc1_meter.avg, loss_meter.avg

if __name__ == '__main__':
    args = parse_option()
    MODEL_TO_RUN = args.model_to_run
    
    # Load model-specific config via models helper
    model_config_data = get_model_config(MODEL_TO_RUN)
    config = Config(model_config_data)
    
    torch.cuda.set_device(0)
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    logger = create_logger(output_dir=config.OUTPUT, name=config.MODEL.NAME)
    main(config, logger)

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def build_loader(config):
    # Enhanced augmentation to reduce overfitting
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(config.DATA.DATA_PATH, "train"), transform=train_transform)
    eval_dataset = datasets.ImageFolder(os.path.join(config.DATA.DATA_PATH, "validation"), transform=eval_transform)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.DATA.BATCH_SIZE, 
        shuffle=True,
        num_workers=config.DATA.NUM_WORKERS, 
        pin_memory=config.DATA.PIN_MEMORY,
    )
    
    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=config.DATA.BATCH_SIZE, 
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS, 
        pin_memory=config.DATA.PIN_MEMORY,
    )

    return train_dataset, eval_dataset, train_loader, eval_loader

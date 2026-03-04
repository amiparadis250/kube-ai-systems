#!/usr/bin/env python3
"""
Local Training Script for Kube-AI Animal Detection
Modified version of code.py without moxing dependencies
"""

import argparse
import os
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import logging
import time
from collections import defaultdict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import model and dataset from code.py
from code import VOCAnimalDataset, SimpleDetectionModel, collate_fn, train_one_epoch

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Animal Detection Local Training')
    parser.add_argument('--data_url', type=str, default='./data',
                       help='Path to training data')
    parser.add_argument('--train_url', type=str, default='./models',
                       help='Path to save model')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--num_classes', type=int, default=5,
                       help='Number of animal classes')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.train_url, exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Log all parameters
    logger.info("="*50)
    logger.info("Starting Animal Detection Local Training")
    logger.info("="*50)
    logger.info(f"Data Path: {args.data_url}")
    logger.info(f"Model Save Path: {args.train_url}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Learning Rate: {args.lr}")
    logger.info(f"Number of Classes: {args.num_classes}")
    logger.info("="*50)
    
    try:
        # Create dataset and dataloader
        logger.info("Creating dataset...")
        
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Create dataset
        dataset = VOCAnimalDataset(
            root_dir=args.data_url,
            transform=transform,
            class_list=['dog', 'cat', 'bird', 'horse', 'cow']
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 for Windows compatibility
            collate_fn=collate_fn,
            pin_memory=False
        )
        
        logger.info(f"Dataset created with {len(dataset)} images")
        logger.info(f"Dataloader has {len(dataloader)} batches")
        
        # Initialize model
        logger.info("Initializing model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        model = SimpleDetectionModel(num_classes=args.num_classes)
        model = model.to(device)
        
        # Loss functions and optimizer
        cls_criterion = nn.CrossEntropyLoss()
        bbox_criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        
        # Resume from checkpoint if requested
        start_epoch = 0
        checkpoint_path = os.path.join(args.train_url, 'checkpoint.pth')
        
        if args.resume and os.path.exists(checkpoint_path):
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            logger.info(f"Resumed from epoch {start_epoch}")
        
        # Training loop
        logger.info("Starting training...")
        training_start = time.time()
        
        for epoch in range(start_epoch, args.epochs):
            epoch_start = time.time()
            
            # Train
            avg_loss, avg_cls, avg_bbox = train_one_epoch(
                model, dataloader, optimizer, cls_criterion, bbox_criterion, device, epoch
            )
            
            # Update scheduler
            scheduler.step()
            
            epoch_time = time.time() - epoch_start
            
            # Log epoch results
            logger.info(f"Epoch {epoch+1}/{args.epochs} completed | "
                       f"Time: {epoch_time:.2f}s | "
                       f"Avg Loss: {avg_loss:.4f} | "
                       f"Avg Cls: {avg_cls:.4f} | "
                       f"Avg BBox: {avg_bbox:.4f}")
            
            # Save checkpoint every 2 epochs
            if (epoch + 1) % 2 == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                }
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Checkpoint saved at epoch {epoch+1}")
        
        total_time = time.time() - training_start
        logger.info(f"Training completed in {total_time/60:.2f} minutes!")
        
        # Save final model
        logger.info("Saving final model...")
        
        # Save PyTorch model
        torch.save(model.state_dict(), os.path.join(args.train_url, 'model.pth'))
        
        # Save with metadata
        torch.save({
            'model_state_dict': model.state_dict(),
            'num_classes': args.num_classes,
            'class_list': dataset.class_list,
            'model_architecture': 'SimpleDetectionModel',
        }, os.path.join(args.train_url, 'model_with_metadata.pth'))
        
        # Save configuration
        with open(os.path.join(args.train_url, 'config.txt'), 'w') as f:
            f.write(f"num_classes: {args.num_classes}\n")
            f.write(f"class_list: {dataset.class_list}\n")
            f.write(f"epochs: {args.epochs}\n")
            f.write(f"batch_size: {args.batch_size}\n")
            f.write(f"learning_rate: {args.lr}\n")
        
        logger.info(f"Model saved to {args.train_url}")
        logger.info("="*50)
        logger.info("LOCAL TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()
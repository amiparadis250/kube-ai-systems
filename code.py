 # ============================================
# train.py - Animal Detection with PyTorch
# Complete code for ModelArts training job
# ============================================

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
import moxing as mox
import logging
import time
from collections import defaultdict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/cache/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================
# 1. VOC Dataset Loader
# ============================================
class VOCAnimalDataset(Dataset):
    """Load PASCAL VOC format for animal detection"""
    
    def __init__(self, root_dir, transform=None, class_list=None):
        """
        Args:
            root_dir: Path to VOC data (contains JPEGImages/ and Annotations/)
            transform: Image transforms
            class_list: List of animal classes to detect
        """
        self.root_dir = root_dir
        self.transform = transform
        self.class_list = class_list or ['dog', 'cat', 'bird', 'horse', 'cow', 'sheep', 'elephant', 'zebra', 'giraffe']
        
        # Find all images and their annotations
        self.images = []
        self.annotations = []
        
        # Check if we have standard VOC structure
        img_dir = os.path.join(root_dir, 'JPEGImages')
        ann_dir = os.path.join(root_dir, 'Annotations')
        
        if os.path.exists(img_dir) and os.path.exists(ann_dir):
            # Standard VOC structure
            for img_file in os.listdir(img_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_name = os.path.splitext(img_file)[0]
                    xml_file = os.path.join(ann_dir, img_name + '.xml')
                    if os.path.exists(xml_file):
                        self.images.append(os.path.join(img_dir, img_file))
                        self.annotations.append(xml_file)
        else:
            # Flat structure (images and XMLs together)
            for file in os.listdir(root_dir):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_name = os.path.splitext(file)[0]
                    xml_file = os.path.join(root_dir, img_name + '.xml')
                    if os.path.exists(xml_file):
                        self.images.append(os.path.join(root_dir, file))
                        self.annotations.append(xml_file)
        
        logger.info(f"Found {len(self.images)} valid image-annotation pairs")
        
        # Create class to id mapping
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_list)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Get original image size for normalization
        orig_width, orig_height = image.size
        
        # Load annotation
        xml_path = self.annotations[idx]
        boxes, labels = self.parse_voc_annotation(xml_path, orig_width, orig_height)
        
        # Apply transforms to image
        if self.transform:
            image = self.transform(image)
        
        # Convert to tensors
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
        labels = torch.tensor(labels, dtype=torch.long) if labels else torch.zeros(0, dtype=torch.long)
        
        return {
            'image': image,
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'orig_size': torch.tensor([orig_height, orig_width])
        }
    
    def parse_voc_annotation(self, xml_file, img_width, img_height):
        """Parse VOC XML and return normalized boxes and labels"""
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        boxes = []
        labels = []
        
        for obj in root.findall('object'):
            # Get class name
            class_name = obj.find('name').text.lower()
            
            # Skip if not in our class list
            if class_name not in self.class_to_idx:
                continue
            
            class_id = self.class_to_idx[class_name]
            
            # Get bounding box
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # Normalize to [0, 1]
            xmin /= img_width
            ymin /= img_height
            xmax /= img_width
            ymax /= img_height
            
            # Convert to [x_center, y_center, width, height] format (optional)
            # x_center = (xmin + xmax) / 2
            # y_center = (ymin + ymax) / 2
            # width = xmax - xmin
            # height = ymax - ymin
            # boxes.append([x_center, y_center, width, height])
            
            # Keep as [xmin, ymin, xmax, ymax]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(class_id)
        
        return boxes, labels

# ============================================
# 2. Custom collate function for variable size
# ============================================
def collate_fn(batch):
    """Custom collate function for detection datasets"""
    images = torch.stack([item['image'] for item in batch])
    boxes = [item['boxes'] for item in batch]
    labels = [item['labels'] for item in batch]
    image_ids = torch.cat([item['image_id'] for item in batch])
    orig_sizes = torch.stack([item['orig_size'] for item in batch])
    
    return {
        'images': images,
        'boxes': boxes,
        'labels': labels,
        'image_ids': image_ids,
        'orig_sizes': orig_sizes
    }

# ============================================
# 3. Simple Detection Model (Faster R-CNN simplified)
# ============================================
class SimpleDetectionModel(nn.Module):
    """Simple CNN for detection - Replace with YOLO/SSD/Faster R-CNN as needed"""
    
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
        # Backbone (simplified - use pretrained in production)
        self.backbone = nn.Sequential(
            # Conv block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        # Detection heads
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
        
        self.bbox_regressor = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 4),
            nn.Sigmoid()  # Output in [0, 1] range
        )
        
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        
        # Predict class and bbox
        class_logits = self.classifier(features)
        bbox_preds = self.bbox_regressor(features)
        
        return class_logits, bbox_preds

# ============================================
# 4. Training Function
# ============================================
def train_one_epoch(model, dataloader, optimizer, cls_criterion, bbox_criterion, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    cls_loss_total = 0
    bbox_loss_total = 0
    
    for batch_idx, batch in enumerate(dataloader):
        images = batch['images'].to(device)
        boxes = batch['boxes']
        labels = batch['labels']
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        class_logits, bbox_preds = model(images)
        
        # Calculate losses (simplified - needs proper matching in production)
        batch_loss = 0
        batch_cls_loss = 0
        batch_bbox_loss = 0
        
        for i in range(len(images)):
            if len(labels[i]) > 0:
                # Classification loss (use first object for simplicity)
                # In production, you need proper matching!
                target_label = labels[i][0].to(device)
                cls_loss = cls_criterion(class_logits[i:i+1], target_label.unsqueeze(0))
                
                # BBox loss
                target_bbox = boxes[i][0].to(device)
                bbox_loss = bbox_criterion(bbox_preds[i], target_bbox)
                
                batch_loss += cls_loss + bbox_loss
                batch_cls_loss += cls_loss.item()
                batch_bbox_loss += bbox_loss.item()
            else:
                # No objects - only background class
                target_label = torch.tensor([model.num_classes - 1]).to(device)  # Background
                cls_loss = cls_criterion(class_logits[i:i+1], target_label)
                batch_loss += cls_loss
                batch_cls_loss += cls_loss.item()
        
        # Backward pass
        batch_loss.backward()
        optimizer.step()
        
        # Update totals
        total_loss += batch_loss.item()
        cls_loss_total += batch_cls_loss
        bbox_loss_total += batch_bbox_loss
        
        # Log every 10 batches
        if batch_idx % 10 == 0:
            logger.info(f'Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | '
                       f'Loss: {batch_loss.item():.4f} | '
                       f'Cls: {batch_cls_loss:.4f} | '
                       f'BBox: {batch_bbox_loss:.4f}')
    
    avg_loss = total_loss / len(dataloader)
    avg_cls = cls_loss_total / len(dataloader)
    avg_bbox = bbox_loss_total / len(dataloader)
    
    return avg_loss, avg_cls, avg_bbox

# ============================================
# 5. Main Function
# ============================================
def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Animal Detection Training')
    parser.add_argument('--data_url', type=str, required=True,
                       help='OBS path to training data')
    parser.add_argument('--train_url', type=str, required=True,
                       help='OBS path to save model')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--num_classes', type=int, default=5,
                       help='Number of animal classes')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Log all parameters
    logger.info("="*50)
    logger.info("Starting Animal Detection Training")
    logger.info("="*50)
    logger.info(f"Data URL: {args.data_url}")
    logger.info(f"Train URL: {args.train_url}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Learning Rate: {args.lr}")
    logger.info(f"Number of Classes: {args.num_classes}")
    logger.info("="*50)
    
    try:
        # ======================================
        # 1. Download data from OBS
        # ======================================
        local_data_path = '/cache/data'
        logger.info(f"Downloading data from {args.data_url} to {local_data_path}")
        
        if not os.path.exists(local_data_path):
            os.makedirs(local_data_path)
        
        mox.file.copy_parallel(args.data_url, local_data_path)
        logger.info("Data download complete!")
        
        # ======================================
        # 2. Create dataset and dataloader
        # ======================================
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
            root_dir=local_data_path,
            transform=transform,
            class_list=['dog', 'cat', 'bird', 'horse', 'cow']  # UPDATE WITH YOUR ANIMALS!
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        logger.info(f"Dataset created with {len(dataset)} images")
        logger.info(f"Dataloader has {len(dataloader)} batches")
        
        # ======================================
        # 3. Initialize model
        # ======================================
        logger.info("Initializing model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        model = SimpleDetectionModel(num_classes=args.num_classes)
        model = model.to(device)
        
        # Loss functions and optimizer
        cls_criterion = nn.CrossEntropyLoss()
        bbox_criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        
        # ======================================
        # 4. Resume from checkpoint if requested
        # ======================================
        start_epoch = 0
        checkpoint_path = '/cache/checkpoint.pth'
        
        if args.resume and os.path.exists(checkpoint_path):
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            logger.info(f"Resumed from epoch {start_epoch}")
        
        # ======================================
        # 5. Training loop
        # ======================================
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
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                }
                torch.save(checkpoint, '/cache/checkpoint.pth')
                logger.info(f"Checkpoint saved at epoch {epoch+1}")
        
        total_time = time.time() - training_start
        logger.info(f"Training completed in {total_time/60:.2f} minutes!")
        
        # ======================================
        # 6. Save final model
        # ======================================
        logger.info("Saving final model...")
        
        # Save model in multiple formats for compatibility
        model_path = '/cache/model'
        os.makedirs(model_path, exist_ok=True)
        
        # Save PyTorch model
        torch.save(model.state_dict(), os.path.join(model_path, 'model.pth'))
        
        # Save with metadata
        torch.save({
            'model_state_dict': model.state_dict(),
            'num_classes': args.num_classes,
            'class_list': dataset.class_list,
            'model_architecture': 'SimpleDetectionModel',
        }, os.path.join(model_path, 'model_with_metadata.pth'))
        
        # Save scripted model for deployment
        scripted_model = torch.jit.script(model)
        scripted_model.save(os.path.join(model_path, 'model_scripted.pt'))
        
        # Save configuration
        with open(os.path.join(model_path, 'config.txt'), 'w') as f:
            f.write(f"num_classes: {args.num_classes}\n")
            f.write(f"class_list: {dataset.class_list}\n")
            f.write(f"epochs: {args.epochs}\n")
            f.write(f"batch_size: {args.batch_size}\n")
            f.write(f"learning_rate: {args.lr}\n")
        
        logger.info(f"Model saved locally to {model_path}")
        
        # ======================================
        # 7. Upload to OBS
        # ======================================
        logger.info(f"Uploading model to {args.train_url}")
        mox.file.copy_parallel(model_path, args.train_url)
        logger.info("Upload complete!")
        
        # ======================================
        # 8. Save training log
        # ======================================
        log_path = '/cache/training.log'
        if os.path.exists(log_path):
            mox.file.copy(log_path, os.path.join(args.train_url, 'training.log'))
        
        logger.info("="*50)
        logger.info("TRAINING JOB COMPLETED SUCCESSFULLY!")
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()
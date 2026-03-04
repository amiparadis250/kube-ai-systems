#!/usr/bin/env python3
"""
MindSpore Animal Detection Training Script
Kube-AI System using MindSpore framework
"""

import argparse
import os
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter
from mindspore.dataset import GeneratorDataset
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
import logging
import time

# Set MindSpore context
ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")  # Change to "GPU" if available

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

class VOCAnimalDataset:
    """VOC format dataset for MindSpore"""
    
    def __init__(self, root_dir, class_list=None):
        self.root_dir = root_dir
        self.class_list = class_list or ['dog', 'cat', 'bird', 'horse', 'cow']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_list)}
        
        # Find all images and annotations
        self.images = []
        self.annotations = []
        
        img_dir = os.path.join(root_dir, 'JPEGImages')
        ann_dir = os.path.join(root_dir, 'Annotations')
        
        if os.path.exists(img_dir) and os.path.exists(ann_dir):
            for img_file in os.listdir(img_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_name = os.path.splitext(img_file)[0]
                    xml_file = os.path.join(ann_dir, img_name + '.xml')
                    if os.path.exists(xml_file):
                        self.images.append(os.path.join(img_dir, img_file))
                        self.annotations.append(xml_file)
        
        logger.info(f"Found {len(self.images)} valid image-annotation pairs")
    
    def __getitem__(self, index):
        # Load image
        img_path = self.images[index]
        image = Image.open(img_path).convert('RGB')
        orig_width, orig_height = image.size
        
        # Resize image
        image = image.resize((224, 224))
        image = np.array(image).astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # HWC to CHW
        
        # Load annotation
        xml_path = self.annotations[index]
        boxes, labels = self.parse_voc_annotation(xml_path, orig_width, orig_height)
        
        # Use first object for simplicity
        if len(labels) > 0:
            label = labels[0]
            bbox = boxes[0]
        else:
            label = 0  # Default class
            bbox = [0.0, 0.0, 1.0, 1.0]  # Default bbox
        
        return image, np.array(label, dtype=np.int32), np.array(bbox, dtype=np.float32)
    
    def __len__(self):
        return len(self.images)
    
    def parse_voc_annotation(self, xml_file, img_width, img_height):
        """Parse VOC XML annotation"""
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        boxes = []
        labels = []
        
        for obj in root.findall('object'):
            class_name = obj.find('name').text.lower()
            if class_name not in self.class_to_idx:
                continue
            
            class_id = self.class_to_idx[class_name]
            
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text) / img_width
            ymin = float(bbox.find('ymin').text) / img_height
            xmax = float(bbox.find('xmax').text) / img_width
            ymax = float(bbox.find('ymax').text) / img_height
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(class_id)
        
        return boxes, labels

class SimpleDetectionModel(nn.Cell):
    """Simple Detection Model using MindSpore"""
    
    def __init__(self, num_classes):
        super(SimpleDetectionModel, self).__init__()
        self.num_classes = num_classes
        
        # Backbone CNN
        self.backbone = nn.SequentialCell([
            # Conv block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1, pad_mode='pad'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1, pad_mode='pad'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1, pad_mode='pad'),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1, pad_mode='pad'),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ])
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Classification head
        self.classifier = nn.SequentialCell([
            nn.Dense(512 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(keep_prob=0.5),
            nn.Dense(1024, num_classes)
        ])
        
        # Bounding box regression head
        self.bbox_regressor = nn.SequentialCell([
            nn.Dense(512 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(keep_prob=0.5),
            nn.Dense(1024, 4),
            nn.Sigmoid()
        ])
        
        # Flatten operation
        self.flatten = nn.Flatten()
    
    def construct(self, x):
        # Extract features
        features = self.backbone(x)
        features = self.global_pool(features)
        features = self.flatten(features)
        
        # Predict class and bbox
        class_logits = self.classifier(features)
        bbox_preds = self.bbox_regressor(features)
        
        return class_logits, bbox_preds

class DetectionLoss(nn.Cell):
    """Combined loss for detection"""
    
    def __init__(self):
        super(DetectionLoss, self).__init__()
        self.cls_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        self.bbox_loss = nn.MSELoss(reduction='mean')
    
    def construct(self, cls_pred, bbox_pred, cls_target, bbox_target):
        cls_loss = self.cls_loss(cls_pred, cls_target)
        bbox_loss = self.bbox_loss(bbox_pred, bbox_target)
        total_loss = cls_loss + bbox_loss
        return total_loss

def create_dataset(data_path, batch_size=4, shuffle=True):
    """Create MindSpore dataset"""
    dataset_generator = VOCAnimalDataset(data_path)
    
    dataset = GeneratorDataset(
        dataset_generator,
        column_names=["image", "label", "bbox"],
        shuffle=shuffle
    )
    
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset

def main():
    parser = argparse.ArgumentParser(description='MindSpore Animal Detection Training')
    parser.add_argument('--data_url', type=str, default='./data', help='Path to training data')
    parser.add_argument('--train_url', type=str, default='./models', help='Path to save model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.train_url, exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    logger.info("="*50)
    logger.info("Starting MindSpore Animal Detection Training")
    logger.info("="*50)
    logger.info(f"Data Path: {args.data_url}")
    logger.info(f"Model Save Path: {args.train_url}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Learning Rate: {args.lr}")
    logger.info(f"Number of Classes: {args.num_classes}")
    logger.info("="*50)
    
    try:
        # Create dataset
        logger.info("Creating dataset...")
        train_dataset = create_dataset(args.data_url, args.batch_size, shuffle=True)
        
        # Create model
        logger.info("Creating model...")
        model = SimpleDetectionModel(num_classes=args.num_classes)
        
        # Define loss and optimizer
        loss_fn = DetectionLoss()
        optimizer = nn.Adam(model.trainable_params(), learning_rate=args.lr)
        
        # Create training model
        train_model = Model(model, loss_fn, optimizer)
        
        # Callbacks
        config_ck = CheckpointConfig(save_checkpoint_steps=10, keep_checkpoint_max=5)
        ckpoint_cb = ModelCheckpoint(prefix="animal_detection", directory=args.train_url, config=config_ck)
        loss_cb = LossMonitor()
        
        # Train
        logger.info("Starting training...")
        start_time = time.time()
        
        train_model.train(args.epochs, train_dataset, callbacks=[ckpoint_cb, loss_cb])
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time/60:.2f} minutes!")
        
        # Save final model
        logger.info("Saving final model...")
        ms.save_checkpoint(model, os.path.join(args.train_url, "final_model.ckpt"))
        
        # Save model info
        with open(os.path.join(args.train_url, 'model_info.txt'), 'w') as f:
            f.write(f"Framework: MindSpore\n")
            f.write(f"Model: SimpleDetectionModel\n")
            f.write(f"Classes: {args.num_classes}\n")
            f.write(f"Epochs: {args.epochs}\n")
            f.write(f"Batch Size: {args.batch_size}\n")
            f.write(f"Learning Rate: {args.lr}\n")
        
        logger.info("="*50)
        logger.info("MINDSPORE TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()
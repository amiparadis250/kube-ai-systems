#!/usr/bin/env python3
"""
KUBE-AI: Aerial Intelligence for Livestock & Wildlife Detection
MindSpore implementation for real-time aerial monitoring
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
ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/kube_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AerialAnimalDataset:
    """Aerial imagery dataset for KUBE Platform"""
    
    def __init__(self, root_dir, class_list=None):
        self.root_dir = root_dir
        # KUBE Platform animal classes for African context
        self.class_list = class_list or [
            'cattle', 'goat', 'sheep', 'elephant', 'zebra', 
            'giraffe', 'buffalo', 'antelope', 'lion', 'leopard'
        ]
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
        
        logger.info(f"🐾 KUBE-AI: Found {len(self.images)} aerial images for training")
    
    def __getitem__(self, index):
        # Load aerial image
        img_path = self.images[index]
        image = Image.open(img_path).convert('RGB')
        orig_width, orig_height = image.size
        
        # Resize for model input (aerial images need different preprocessing)
        image = image.resize((224, 224))
        image = np.array(image).astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # HWC to CHW
        
        # Load annotation
        xml_path = self.annotations[index]
        boxes, labels = self.parse_voc_annotation(xml_path, orig_width, orig_height)
        
        # Use first detected animal for training
        if len(labels) > 0:
            label = labels[0]
            bbox = boxes[0]
        else:
            label = 0  # Default to cattle
            bbox = [0.0, 0.0, 1.0, 1.0]
        
        return image, np.array(label, dtype=np.int32), np.array(bbox, dtype=np.float32)
    
    def __len__(self):
        return len(self.images)
    
    def parse_voc_annotation(self, xml_file, img_width, img_height):
        """Parse VOC XML for aerial animal detection"""
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

class KubeAIModel(nn.Cell):
    """KUBE-AI: Aerial Animal Detection Model"""
    
    def __init__(self, num_classes):
        super(KubeAIModel, self).__init__()
        self.num_classes = num_classes
        
        # Enhanced backbone for aerial imagery
        self.backbone = nn.SequentialCell([
            # Block 1 - Edge detection for aerial views
            nn.Conv2d(3, 64, kernel_size=3, padding=1, pad_mode='pad'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2 - Pattern recognition
            nn.Conv2d(64, 128, kernel_size=3, padding=1, pad_mode='pad'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3 - Feature extraction
            nn.Conv2d(128, 256, kernel_size=3, padding=1, pad_mode='pad'),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4 - High-level features
            nn.Conv2d(256, 512, kernel_size=3, padding=1, pad_mode='pad'),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 5 - Deep features for small objects
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, pad_mode='pad'),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        ])
        
        # Animal classification head
        self.classifier = nn.SequentialCell([
            nn.Dense(1024 * 7 * 7, 2048),
            nn.ReLU(),
            nn.Dropout(keep_prob=0.5),
            nn.Dense(2048, 1024),
            nn.ReLU(),
            nn.Dropout(keep_prob=0.5),
            nn.Dense(1024, num_classes)
        ])
        
        # Bounding box regression for animal location
        self.bbox_regressor = nn.SequentialCell([
            nn.Dense(1024 * 7 * 7, 2048),
            nn.ReLU(),
            nn.Dropout(keep_prob=0.5),
            nn.Dense(2048, 1024),
            nn.ReLU(),
            nn.Dropout(keep_prob=0.5),
            nn.Dense(1024, 4),
            nn.Sigmoid()
        ])
        
        self.flatten = nn.Flatten()
    
    def construct(self, x):
        # Extract aerial features
        features = self.backbone(x)
        features = self.flatten(features)
        
        # Predict animal type and location
        class_logits = self.classifier(features)
        bbox_preds = self.bbox_regressor(features)
        
        return class_logits, bbox_preds

class KubeDetectionLoss(nn.Cell):
    """KUBE-AI Detection Loss Function"""
    
    def __init__(self):
        super(KubeDetectionLoss, self).__init__()
        self.cls_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        self.bbox_loss = nn.MSELoss(reduction='mean')
    
    def construct(self, cls_pred, bbox_pred, cls_target, bbox_target):
        cls_loss = self.cls_loss(cls_pred, cls_target)
        bbox_loss = self.bbox_loss(bbox_pred, bbox_target)
        # Weight bbox loss higher for precise localization
        total_loss = cls_loss + 2.0 * bbox_loss
        return total_loss

def create_dataset(data_path, batch_size=4, shuffle=True):
    """Create KUBE-AI dataset"""
    dataset_generator = AerialAnimalDataset(data_path)
    
    dataset = GeneratorDataset(
        dataset_generator,
        column_names=["image", "label", "bbox"],
        shuffle=shuffle
    )
    
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset

def main():
    parser = argparse.ArgumentParser(description='KUBE-AI: Aerial Animal Detection Training')
    parser.add_argument('--data_url', type=str, default='./data', help='Path to aerial imagery data')
    parser.add_argument('--train_url', type=str, default='./models', help='Path to save KUBE-AI model')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of animal classes')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.train_url, exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    logger.info("="*60)
    logger.info("🚁 KUBE-AI: Aerial Intelligence Training Started")
    logger.info("🌍 Mission: Protect Africa's Wildlife & Livestock")
    logger.info("="*60)
    logger.info(f"📊 Data Path: {args.data_url}")
    logger.info(f"💾 Model Save Path: {args.train_url}")
    logger.info(f"🔄 Epochs: {args.epochs}")
    logger.info(f"📦 Batch Size: {args.batch_size}")
    logger.info(f"📈 Learning Rate: {args.lr}")
    logger.info(f"🐾 Animal Classes: {args.num_classes}")
    logger.info("="*60)
    
    try:
        # Create aerial dataset
        logger.info("📡 Loading aerial imagery dataset...")
        train_dataset = create_dataset(args.data_url, args.batch_size, shuffle=True)
        
        # Create KUBE-AI model
        logger.info("🧠 Initializing KUBE-AI detection model...")
        model = KubeAIModel(num_classes=args.num_classes)
        
        # Define loss and optimizer
        loss_fn = KubeDetectionLoss()
        optimizer = nn.Adam(model.trainable_params(), learning_rate=args.lr)
        
        # Create training model
        train_model = Model(model, loss_fn, optimizer)
        
        # Callbacks for monitoring
        config_ck = CheckpointConfig(save_checkpoint_steps=20, keep_checkpoint_max=10)
        ckpoint_cb = ModelCheckpoint(prefix="kube_ai", directory=args.train_url, config=config_ck)
        loss_cb = LossMonitor()
        
        # Train KUBE-AI
        logger.info("🚀 Starting KUBE-AI training...")
        start_time = time.time()
        
        train_model.train(args.epochs, train_dataset, callbacks=[ckpoint_cb, loss_cb])
        
        training_time = time.time() - start_time
        logger.info(f"✅ KUBE-AI training completed in {training_time/60:.2f} minutes!")
        
        # Save final model
        logger.info("💾 Saving KUBE-AI model...")
        ms.save_checkpoint(model, os.path.join(args.train_url, "kube_ai_final.ckpt"))
        
        # Save model metadata
        with open(os.path.join(args.train_url, 'kube_ai_info.txt'), 'w') as f:
            f.write("KUBE-AI: Aerial Intelligence for Africa\n")
            f.write("="*40 + "\n")
            f.write(f"Framework: MindSpore\n")
            f.write(f"Model: KubeAIModel (Enhanced CNN)\n")
            f.write(f"Purpose: Livestock & Wildlife Detection\n")
            f.write(f"Classes: {args.num_classes}\n")
            f.write(f"Epochs: {args.epochs}\n")
            f.write(f"Batch Size: {args.batch_size}\n")
            f.write(f"Learning Rate: {args.lr}\n")
            f.write(f"Training Time: {training_time/60:.2f} minutes\n")
            f.write("\nCapabilities:\n")
            f.write("- Real-time aerial animal detection\n")
            f.write("- Livestock counting and tracking\n")
            f.write("- Wildlife census and monitoring\n")
            f.write("- Anti-poaching surveillance\n")
        
        logger.info("="*60)
        logger.info("🎉 KUBE-AI TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("🌍 Ready to protect Africa's animals!")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"❌ KUBE-AI training failed: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
MindSpore Animal Detection Inference Script
"""

import argparse
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
import json

# Set MindSpore context
ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

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

class AnimalDetector:
    """MindSpore Animal Detector"""
    
    def __init__(self, model_path, num_classes=5):
        self.num_classes = num_classes
        self.class_list = ['dog', 'cat', 'bird', 'horse', 'cow']
        
        # Create model
        self.model = SimpleDetectionModel(num_classes=num_classes)
        
        # Load checkpoint
        param_dict = ms.load_checkpoint(model_path)
        ms.load_param_into_net(self.model, param_dict)
        self.model.set_train(False)
        
        print(f"✅ Model loaded from {model_path}")
        print(f"📋 Classes: {self.class_list}")
    
    def preprocess_image(self, image_path):
        """Preprocess image for inference"""
        image = Image.open(image_path).convert('RGB')
        orig_width, orig_height = image.size
        
        # Resize and normalize
        image_resized = image.resize((224, 224))
        image_array = np.array(image_resized).astype(np.float32) / 255.0
        image_array = np.transpose(image_array, (2, 0, 1))  # HWC to CHW
        image_tensor = Tensor(image_array[np.newaxis, :], ms.float32)
        
        return image_tensor, image, (orig_width, orig_height)
    
    def predict(self, image_path):
        """Run prediction on image"""
        # Preprocess
        image_tensor, orig_image, orig_size = self.preprocess_image(image_path)
        
        # Inference
        class_logits, bbox_preds = self.model(image_tensor)
        
        # Get predictions
        class_probs = nn.Softmax(axis=1)(class_logits)
        confidence = float(np.max(class_probs.asnumpy()))
        predicted_class = int(np.argmax(class_probs.asnumpy()))
        bbox = bbox_preds.asnumpy()[0]
        
        # Convert normalized bbox to pixel coordinates
        orig_width, orig_height = orig_size
        xmin = int(bbox[0] * orig_width)
        ymin = int(bbox[1] * orig_height)
        xmax = int(bbox[2] * orig_width)
        ymax = int(bbox[3] * orig_height)
        
        result = {
            'class': self.class_list[predicted_class] if predicted_class < len(self.class_list) else f'class_{predicted_class}',
            'confidence': confidence,
            'bbox': [xmin, ymin, xmax, ymax],
            'image_size': [orig_width, orig_height]
        }
        
        return result, orig_image
    
    def visualize(self, image, result, output_path):
        """Draw bounding box and label on image"""
        draw = ImageDraw.Draw(image)
        
        # Draw bbox
        bbox = result['bbox']
        draw.rectangle(bbox, outline='red', width=3)
        
        # Draw label
        label = f"{result['class']}: {result['confidence']:.2f}"
        draw.text((bbox[0], bbox[1] - 20), label, fill='red')
        
        # Save
        image.save(output_path)
        print(f"🖼️ Visualization saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='MindSpore Animal Detection Inference')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--output_path', type=str, default='output.jpg', help='Path to save output')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = AnimalDetector(
        model_path=args.model_path,
        num_classes=args.num_classes
    )
    
    # Run prediction
    print(f"\n🔍 Processing: {args.image_path}")
    result, orig_image = detector.predict(args.image_path)
    
    # Print results
    print("\n" + "="*50)
    print("🐾 DETECTION RESULTS")
    print("="*50)
    print(f"Class: {result['class']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Bounding Box: {result['bbox']}")
    print("="*50)
    
    # Visualize
    detector.visualize(orig_image, result, args.output_path)
    
    # Save results to JSON
    json_path = args.output_path.replace('.jpg', '.json').replace('.png', '.json')
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=4)
    print(f"📄 Results saved to {json_path}")

if __name__ == '__main__':
    main()
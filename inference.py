# ============================================
# inference.py - Animal Detection Inference
# Run predictions on new images
# ============================================

import argparse
import os
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import json

# Import model from code.py
from code import SimpleDetectionModel

class AnimalDetector:
    """Inference class for animal detection"""
    
    def __init__(self, model_path, num_classes=5, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        
        # Load model
        self.model = SimpleDetectionModel(num_classes=num_classes)
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.class_list = checkpoint.get('class_list', [f'class_{i}' for i in range(num_classes)])
        else:
            self.model.load_state_dict(checkpoint)
            self.class_list = [f'class_{i}' for i in range(num_classes)]
        
        self.model.to(self.device)
        self.model.eval()
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Classes: {self.class_list}")
    
    def predict(self, image_path, confidence_threshold=0.5):
        """Run prediction on a single image"""
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        orig_width, orig_height = image.size
        
        # Transform
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            class_logits, bbox_preds = self.model(image_tensor)
            
            # Get predictions
            probs = torch.softmax(class_logits, dim=1)
            confidence, predicted_class = torch.max(probs, 1)
            
            confidence = confidence.item()
            predicted_class = predicted_class.item()
            bbox = bbox_preds[0].cpu().numpy()
        
        # Convert normalized bbox to pixel coordinates
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
        
        return result
    
    def visualize(self, image_path, output_path, result):
        """Draw bounding box and label on image"""
        image = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(image)
        
        # Draw bbox
        bbox = result['bbox']
        draw.rectangle(bbox, outline='red', width=3)
        
        # Draw label
        label = f"{result['class']}: {result['confidence']:.2f}"
        draw.text((bbox[0], bbox[1] - 20), label, fill='red')
        
        # Save
        image.save(output_path)
        print(f"Visualization saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Animal Detection Inference')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model (.pth file)')
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--output_path', type=str, default='output.jpg',
                       help='Path to save output image')
    parser.add_argument('--num_classes', type=int, default=5,
                       help='Number of classes')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                       help='Confidence threshold')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = AnimalDetector(
        model_path=args.model_path,
        num_classes=args.num_classes
    )
    
    # Run prediction
    print(f"\nProcessing: {args.image_path}")
    result = detector.predict(args.image_path, args.confidence_threshold)
    
    # Print results
    print("\n" + "="*50)
    print("DETECTION RESULTS")
    print("="*50)
    print(f"Class: {result['class']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Bounding Box: {result['bbox']}")
    print("="*50)
    
    # Visualize
    detector.visualize(args.image_path, args.output_path, result)
    
    # Save results to JSON
    json_path = args.output_path.replace('.jpg', '.json').replace('.png', '.json')
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=4)
    print(f"Results saved to {json_path}")

if __name__ == '__main__':
    main()

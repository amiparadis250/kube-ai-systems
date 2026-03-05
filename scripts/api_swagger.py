from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
from werkzeug.datastructures import FileStorage
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import time
import os

app = Flask(__name__)
api = Api(app, 
    version='1.0', 
    title='KUBE-AI API',
    description='Aerial Intelligence for African Wildlife & Livestock Detection',
    doc='/docs/'
)

ns = api.namespace('kube-ai', description='Animal Detection Operations')

# Swagger models
detection_model = api.model('Detection', {
    'detection_id': fields.String(description='Unique detection ID'),
    'animal_type': fields.String(description='Detected animal type'),
    'confidence': fields.Float(description='Detection confidence (0-1)'),
    'bbox': fields.List(fields.Integer, description='Bounding box [xmin, ymin, xmax, ymax]'),
    'kube_module': fields.String(description='KUBE-Farm or KUBE-Park'),
    'alert_level': fields.String(description='Alert priority level'),
    'inference_time_ms': fields.Float(description='Processing time in milliseconds'),
    'timestamp': fields.String(description='Detection timestamp'),
    'mode': fields.String(description='DEMO or PRODUCTION')
})

upload_parser = api.parser()
upload_parser.add_argument('image', location='files', type=FileStorage, required=True, help='Image file')

class KubeModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256 * 28 * 28, num_classes)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        return self.fc(self.flatten(x))

# Load model (with fallback for demo)
try:
    model = KubeModel()
    # Try balanced model first, then fall back to regular
    try:
        model.load_state_dict(torch.load('../models/kube_pytorch_balanced.pth', map_location='cpu'))
        print("✅ KUBE-AI Balanced model loaded")
    except:
        model.load_state_dict(torch.load('../models/kube_pytorch.pth', map_location='cpu'))
        print("⚠️ Using original model (may be overfitted)")
    
    model.eval()
    MODEL_LOADED = True
except Exception as e:
    model = None
    MODEL_LOADED = False
    print(f"⚠️ Model not found ({e}) - using demo mode")

classes = ['cattle', 'goat', 'sheep', 'elephant', 'zebra', 'giraffe', 'buffalo', 'antelope', 'lion', 'leopard']

@ns.route('/predict')
class Predict(Resource):
    @ns.expect(upload_parser)
    @ns.marshal_with(detection_model)
    @ns.doc('predict_animal')
    def post(self):
        """Detect animals in uploaded aerial image"""
        try:
            if 'image' not in request.files:
                api.abort(400, 'No image provided')
            
            file = request.files['image']
            img = Image.open(file.stream).convert('RGB').resize((224, 224))
            
            start_time = time.time()
            
            if MODEL_LOADED:
                # Use actual model
                img_tensor = torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0
                img_tensor = img_tensor.unsqueeze(0)
                
                with torch.no_grad():
                    outputs = model(img_tensor)
                    probabilities = torch.softmax(outputs, 1)[0]
                    _, predicted = torch.max(outputs, 1)
                    confidence = probabilities[predicted].item()
                
                animal = classes[predicted.item()]
                
                # Debug: Print all class probabilities
                print(f"\n🔍 Model Predictions:")
                for i, (cls, prob) in enumerate(zip(classes, probabilities)):
                    print(f"  {cls}: {prob:.4f} {'← PREDICTED' if i == predicted.item() else ''}")
                print(f"Final: {animal} ({confidence:.4f})\n")
            else:
                # Demo mode - analyze image colors/features for better simulation
                import random
                img_array = np.array(img)
                
                # Simple heuristics based on image properties
                avg_color = np.mean(img_array, axis=(0,1))
                brightness = np.mean(avg_color)
                
                # More realistic prediction based on image characteristics
                if brightness > 150:  # Bright image - likely livestock
                    animal = random.choice(['cattle', 'goat', 'sheep'])
                    confidence = random.uniform(0.85, 0.95)
                elif brightness < 100:  # Dark image - might be wildlife
                    animal = random.choice(['elephant', 'buffalo'])
                    confidence = random.uniform(0.75, 0.90)
                else:  # Medium brightness
                    animal = random.choice(['cattle', 'zebra', 'elephant'])
                    confidence = random.uniform(0.80, 0.92)
                
                # Add some randomness to make it feel more realistic
                if random.random() < 0.1:  # 10% chance of different prediction
                    animal = random.choice(['cattle', 'elephant', 'zebra'])
            
            processing_time = (time.time() - start_time) * 1000
            
            alert_level = "CRITICAL" if animal in ['lion', 'leopard'] else "STANDARD"
            module = "KUBE-Park" if animal in ['elephant', 'zebra', 'giraffe', 'buffalo', 'antelope', 'lion', 'leopard'] else "KUBE-Farm"
            
            return {
                'detection_id': f'kube_{int(time.time())}',
                'animal_type': animal,
                'confidence': round(confidence, 4),
                'bbox': [50, 50, 200, 200],  # Placeholder bbox
                'kube_module': module,
                'alert_level': alert_level,
                'inference_time_ms': round(processing_time, 2),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'mode': 'PRODUCTION' if MODEL_LOADED else 'DEMO'
            }
            
        except Exception as e:
            api.abort(500, str(e))

@ns.route('/health')
class Health(Resource):
    @ns.doc('health_check')
    def get(self):
        """Check API health status"""
        return {'status': 'KUBE-AI API is running', 'version': '1.0'}

if __name__ == '__main__':
    print("🚁 KUBE-AI API Starting...")
    print("📡 Endpoint: http://localhost:5000/predict")
    print("📚 Swagger Docs: http://localhost:5000/docs/")
    app.run(debug=True, host='0.0.0.0', port=5000)
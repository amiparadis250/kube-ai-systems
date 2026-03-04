#!/usr/bin/env python3
"""
Dataset Downloader for Kube-AI Animal Detection
Downloads sample animal images and creates VOC annotations
"""

import os
import requests
import json
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
from xml.dom import minidom

# Sample animal images URLs (free stock photos)
SAMPLE_IMAGES = {
    'dog': [
        'https://images.unsplash.com/photo-1552053831-71594a27632d?w=640',
        'https://images.unsplash.com/photo-1583337130417-3346a1be7dee?w=640',
    ],
    'cat': [
        'https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=640',
        'https://images.unsplash.com/photo-1573865526739-10659fec78a5?w=640',
    ],
    'bird': [
        'https://images.unsplash.com/photo-1444464666168-49d633b86797?w=640',
        'https://images.unsplash.com/photo-1426604966848-d7adac402bff?w=640',
    ],
    'horse': [
        'https://images.unsplash.com/photo-1553284965-83fd3e82fa5a?w=640',
        'https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=640',
    ],
    'cow': [
        'https://images.unsplash.com/photo-1516467508483-a7212febe31a?w=640',
        'https://images.unsplash.com/photo-1572021335469-31706a17aaef?w=640',
    ]
}

def create_directories():
    """Create required directories"""
    os.makedirs('data/JPEGImages', exist_ok=True)
    os.makedirs('data/Annotations', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    print("✅ Directories created")

def download_image(url, filepath):
    """Download image from URL"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(filepath, 'wb') as f:
            f.write(response.content)
        return True
    except Exception as e:
        print(f"❌ Failed to download {url}: {e}")
        return False

def create_voc_annotation(image_path, class_name, bbox, output_path):
    """Create VOC format XML annotation"""
    # Get image dimensions
    with Image.open(image_path) as img:
        width, height = img.size
    
    # Create XML structure
    annotation = ET.Element('annotation')
    
    # Basic info
    ET.SubElement(annotation, 'folder').text = 'JPEGImages'
    ET.SubElement(annotation, 'filename').text = os.path.basename(image_path)
    ET.SubElement(annotation, 'path').text = image_path
    
    # Source
    source = ET.SubElement(annotation, 'source')
    ET.SubElement(source, 'database').text = 'Kube-AI Animal Dataset'
    
    # Size
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = '3'
    
    ET.SubElement(annotation, 'segmented').text = '0'
    
    # Object
    obj = ET.SubElement(annotation, 'object')
    ET.SubElement(obj, 'name').text = class_name
    ET.SubElement(obj, 'pose').text = 'Unspecified'
    ET.SubElement(obj, 'truncated').text = '0'
    ET.SubElement(obj, 'difficult').text = '0'
    
    # Bounding box
    bndbox = ET.SubElement(obj, 'bndbox')
    ET.SubElement(bndbox, 'xmin').text = str(bbox[0])
    ET.SubElement(bndbox, 'ymin').text = str(bbox[1])
    ET.SubElement(bndbox, 'xmax').text = str(bbox[2])
    ET.SubElement(bndbox, 'ymax').text = str(bbox[3])
    
    # Pretty print and save
    rough_string = ET.tostring(annotation, 'unicode')
    reparsed = minidom.parseString(rough_string)
    with open(output_path, 'w') as f:
        f.write(reparsed.toprettyxml(indent="  "))

def download_sample_dataset():
    """Download sample animal dataset"""
    print("🐾 Downloading Kube-AI Animal Dataset...")
    
    create_directories()
    
    image_count = 0
    for class_name, urls in SAMPLE_IMAGES.items():
        print(f"\n📥 Downloading {class_name} images...")
        
        for i, url in enumerate(urls):
            # Download image
            image_filename = f"{class_name}_{i+1:03d}.jpg"
            image_path = f"data/JPEGImages/{image_filename}"
            
            if download_image(url, image_path):
                print(f"  ✅ {image_filename}")
                
                # Create annotation with estimated bounding box
                # (In real scenario, you'd manually annotate or use detection tools)
                with Image.open(image_path) as img:
                    w, h = img.size
                    # Rough center bbox (you should manually adjust these)
                    bbox = [
                        int(w * 0.2),  # xmin
                        int(h * 0.2),  # ymin  
                        int(w * 0.8),  # xmax
                        int(h * 0.8)   # ymax
                    ]
                
                # Create XML annotation
                xml_filename = f"{class_name}_{i+1:03d}.xml"
                xml_path = f"data/Annotations/{xml_filename}"
                create_voc_annotation(image_path, class_name, bbox, xml_path)
                
                image_count += 1
            else:
                print(f"  ❌ Failed: {image_filename}")
    
    print(f"\n🎉 Dataset created with {image_count} images!")
    print("📁 Structure:")
    print("  data/JPEGImages/ - Training images")
    print("  data/Annotations/ - VOC XML annotations")
    print("\n⚠️  IMPORTANT: The bounding boxes are estimated!")
    print("   For better results, manually adjust the bounding boxes in XML files")

if __name__ == "__main__":
    download_sample_dataset()
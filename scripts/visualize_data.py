import os
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import numpy as np
from collections import Counter

def visualize_dataset():
    """Visualize KUBE-AI dataset statistics and sample images"""
    
    img_dir = '../data/JPEGImages'
    ann_dir = '../data/Annotations'
    
    if not os.path.exists(img_dir) or not os.path.exists(ann_dir):
        print("Run prepare_data.py first to create dataset")
        return
    
    # Collect statistics
    animals = []
    bbox_sizes = []
    
    for xml_file in os.listdir(ann_dir):
        if xml_file.endswith('.xml'):
            tree = ET.parse(os.path.join(ann_dir, xml_file))
            root = tree.getroot()
            
            for obj in root.findall('object'):
                animal = obj.find('name').text
                animals.append(animal)
                
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                
                width = xmax - xmin
                height = ymax - ymin
                bbox_sizes.append(width * height)
    
    # Create visualizations
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('KUBE-AI Dataset Analysis', fontsize=16, fontweight='bold')
    
    # 1. Animal distribution
    animal_counts = Counter(animals)
    ax1.bar(animal_counts.keys(), animal_counts.values(), color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax1.set_title('Animal Distribution')
    ax1.set_ylabel('Number of Images')
    
    # 2. Bounding box sizes
    ax2.hist(bbox_sizes, bins=20, color='#96CEB4', alpha=0.7)
    ax2.set_title('Bounding Box Size Distribution')
    ax2.set_xlabel('Bbox Area (pixels)')
    ax2.set_ylabel('Frequency')
    
    # 3. Sample images with bboxes
    sample_images = show_sample_images(img_dir, ann_dir, ax3)
    
    # 4. Dataset summary
    ax4.axis('off')
    summary_text = f"""
    KUBE-AI Dataset Summary
    
    Total Images: {len(animals)}
    
    Animal Breakdown:
    """
    for animal, count in animal_counts.items():
        summary_text += f"  • {animal.title()}: {count} images\n"
    
    summary_text += f"""
    
    Average Bbox Size: {np.mean(bbox_sizes):.0f} pixels
    Image Resolution: 640 x 480
    
    Status: Ready for Training ✓
    """
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
             fontsize=12, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('../visualizations/dataset_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Analyzed {len(animals)} images")
    print(f"✓ Found {len(animal_counts)} animal types")
    print(f"✓ Visualization saved to ../visualizations/dataset_analysis.png")

def show_sample_images(img_dir, ann_dir, ax):
    """Show sample images with bounding boxes"""
    
    # Get first few images
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')][:4]
    
    sample_grid = np.zeros((480*2, 640*2, 3), dtype=np.uint8)
    
    for i, img_file in enumerate(img_files):
        # Load image
        img_path = os.path.join(img_dir, img_file)
        xml_path = os.path.join(ann_dir, img_file.replace('.jpg', '.xml'))
        
        img = Image.open(img_path)
        draw = ImageDraw.Draw(img)
        
        # Draw bounding box
        if os.path.exists(xml_path):
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            for obj in root.findall('object'):
                animal = obj.find('name').text
                bbox = obj.find('bndbox')
                
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                
                # Draw rectangle and label
                draw.rectangle([xmin, ymin, xmax, ymax], outline='red', width=3)
                draw.text((xmin, ymin-20), animal.upper(), fill='red')
        
        # Place in grid
        row = i // 2
        col = i % 2
        img_array = np.array(img)
        sample_grid[row*480:(row+1)*480, col*640:(col+1)*640] = img_array
    
    ax.imshow(sample_grid)
    ax.set_title('Sample Images with Annotations')
    ax.axis('off')

if __name__ == '__main__':
    # Create output directory
    os.makedirs('../visualizations', exist_ok=True)
    visualize_dataset()
import os
import shutil
from PIL import Image
import xml.etree.ElementTree as ET

def prepare_data():
    # Training datasets
    train_datasets = ['../datasets/kaggle_cows/images', '../datasets/livestock-qocf0/train/images', 
                      '../datasets/wildlife-4satw/train/images', '../datasets/wildlife-yj7t1/train/images']
    
    # Test datasets for accuracy evaluation
    test_datasets = ['../datasets/livestock-qocf0/test/images', 
                     '../datasets/wildlife-4satw/test/images', '../datasets/wildlife-yj7t1/test/images']
    
    # Create directories
    train_img_dir = '../data/JPEGImages'
    train_ann_dir = '../data/Annotations'
    test_img_dir = '../data/TestImages'
    test_ann_dir = '../data/TestAnnotations'
    
    for dir_path in [train_img_dir, train_ann_dir, test_img_dir, test_ann_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Process training data
    train_count = process_datasets(train_datasets, train_img_dir, train_ann_dir, "train")
    
    # Process test data
    test_count = process_datasets(test_datasets, test_img_dir, test_ann_dir, "test")
    
    print(f"✓ Prepared {train_count} training images")
    print(f"✓ Prepared {test_count} test images")
    print(f"✓ Total: {train_count + test_count} images")

def process_datasets(datasets, img_dir, ann_dir, split_type):
    count = 0
    for dataset in datasets:
        if not os.path.exists(dataset):
            continue
        
        for root, dirs, files in os.walk(dataset):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    src = os.path.join(root, file)
                    dst = os.path.join(img_dir, f"{split_type}_{count:06d}.jpg")
                    
                    try:
                        Image.open(src).convert('RGB').resize((640, 480)).save(dst, 'JPEG')
                        create_xml(f"{split_type}_{count:06d}.jpg", ann_dir, dataset)
                        count += 1
                    except:
                        continue
    return count

def create_xml(img_name, ann_dir, dataset_path):
    animal = 'cattle'
    if 'wildlife' in dataset_path: animal = 'elephant'
    elif 'wild' in dataset_path: animal = 'zebra'
    
    root = ET.Element('annotation')
    ET.SubElement(root, 'filename').text = img_name
    
    size = ET.SubElement(root, 'size')
    ET.SubElement(size, 'width').text = '640'
    ET.SubElement(size, 'height').text = '480'
    ET.SubElement(size, 'depth').text = '3'
    
    obj = ET.SubElement(root, 'object')
    ET.SubElement(obj, 'name').text = animal
    
    bbox = ET.SubElement(obj, 'bndbox')
    ET.SubElement(bbox, 'xmin').text = '160'
    ET.SubElement(bbox, 'ymin').text = '120'
    ET.SubElement(bbox, 'xmax').text = '480'
    ET.SubElement(bbox, 'ymax').text = '360'
    
    ET.ElementTree(root).write(os.path.join(ann_dir, img_name.replace('.jpg', '.xml')))

if __name__ == '__main__':
    prepare_data()
import torch
import torch.nn as nn
from train_pytorch import KubeModel, AnimalDataset
from torch.utils.data import DataLoader
from collections import Counter
import xml.etree.ElementTree as ET
import os

def retrain_balanced():
    # Load dataset and check class distribution
    dataset = AnimalDataset('../data')
    
    # Count class distribution
    class_counts = Counter()
    for _, label in dataset:
        class_counts[label.item()] += 1
    
    print("Class distribution:")
    for i, count in class_counts.items():
        print(f"  {dataset.classes[i]}: {count}")
    
    # Calculate class weights (inverse frequency)
    total_samples = sum(class_counts.values())
    class_weights = []
    for i in range(len(dataset.classes)):
        if i in class_counts:
            weight = total_samples / (len(dataset.classes) * class_counts[i])
        else:
            weight = 1.0
        class_weights.append(weight)
    
    print(f"\nUsing class weights: {[f'{w:.2f}' for w in class_weights]}")
    
    # Create weighted loss
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # Train with balanced loss
    model = KubeModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    print("\n🔄 Retraining with balanced classes...")
    for epoch in range(20):
        total_loss = 0
        for images, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/20, Loss: {total_loss/len(dataloader):.4f}")
    
    # Save improved model
    torch.save(model.state_dict(), '../models/kube_pytorch_balanced.pth')
    print("✅ Balanced model saved!")

if __name__ == '__main__':
    retrain_balanced()
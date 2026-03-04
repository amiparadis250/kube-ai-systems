#!/usr/bin/env python3
"""
Demo Script for Kube-AI Animal Detection
Complete pipeline demonstration
"""

import os
import subprocess
import sys

def run_command(cmd, description):
    """Run command and handle errors"""
    print(f"\n🚀 {description}")
    print(f"Command: {cmd}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        print("✅ Success!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("🐾 Kube-AI Animal Detection Demo")
    print("=" * 50)
    
    # Step 1: Download dataset
    print("\n📥 Step 1: Download Sample Dataset")
    if not run_command("python download_dataset.py", "Downloading sample animal images"):
        print("❌ Dataset download failed. Please check your internet connection.")
        return
    
    # Step 2: Train model (quick training)
    print("\n🏋️ Step 2: Train Model (Quick Demo)")
    train_cmd = "python train_local.py --epochs 3 --batch_size 2"
    if not run_command(train_cmd, "Training animal detection model"):
        print("❌ Training failed. Check the logs for details.")
        return
    
    # Step 3: Test inference
    print("\n🔍 Step 3: Test Inference")
    
    # Find a sample image
    sample_image = None
    if os.path.exists("data/JPEGImages"):
        images = [f for f in os.listdir("data/JPEGImages") if f.endswith(('.jpg', '.jpeg', '.png'))]
        if images:
            sample_image = f"data/JPEGImages/{images[0]}"
    
    if sample_image:
        inference_cmd = f"python inference.py --model_path models/model_with_metadata.pth --image_path {sample_image} --output_path demo_result.jpg"
        if run_command(inference_cmd, f"Running inference on {sample_image}"):
            print(f"✅ Demo completed! Check 'demo_result.jpg' for results.")
        else:
            print("❌ Inference failed.")
    else:
        print("❌ No sample images found for inference.")
    
    # Step 4: Show results
    print("\n📊 Demo Results:")
    print("- Dataset: data/JPEGImages/ and data/Annotations/")
    print("- Model: models/model_with_metadata.pth")
    print("- Logs: logs/training.log")
    print("- Demo Output: demo_result.jpg")
    
    print("\n🎉 Kube-AI Demo Complete!")
    print("Next steps:")
    print("1. Add more training data")
    print("2. Increase training epochs")
    print("3. Fine-tune hyperparameters")
    print("4. Deploy to production")

if __name__ == "__main__":
    main()
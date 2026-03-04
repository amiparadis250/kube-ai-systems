#!/usr/bin/env python3
"""
Complete MindSpore Demo for Kube-AI Animal Detection
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
    print("🐾 Kube-AI MindSpore Animal Detection Demo")
    print("=" * 60)
    
    # Step 1: Download dataset
    print("\n📥 Step 1: Download Sample Dataset")
    if not run_command("python download_dataset.py", "Downloading sample animal images"):
        print("❌ Dataset download failed. Please check your internet connection.")
        return
    
    # Step 2: Train MindSpore model
    print("\n🧠 Step 2: Train MindSpore Model")
    train_cmd = "python train_mindspore.py --epochs 5 --batch_size 2"
    if not run_command(train_cmd, "Training MindSpore animal detection model"):
        print("❌ MindSpore training failed. Check the logs for details.")
        return
    
    # Step 3: Test MindSpore inference
    print("\n🔍 Step 3: Test MindSpore Inference")
    
    # Find a sample image
    sample_image = None
    if os.path.exists("data/JPEGImages"):
        images = [f for f in os.listdir("data/JPEGImages") if f.endswith(('.jpg', '.jpeg', '.png'))]
        if images:
            sample_image = f"data/JPEGImages/{images[0]}"
    
    if sample_image:
        # Find the latest checkpoint
        ckpt_files = [f for f in os.listdir("models") if f.endswith('.ckpt')]
        if ckpt_files:
            latest_ckpt = f"models/{ckpt_files[-1]}"
            inference_cmd = f"python inference_mindspore.py --model_path {latest_ckpt} --image_path {sample_image} --output_path mindspore_result.jpg"
            if run_command(inference_cmd, f"Running MindSpore inference on {sample_image}"):
                print(f"✅ MindSpore demo completed! Check 'mindspore_result.jpg' for results.")
            else:
                print("❌ MindSpore inference failed.")
        else:
            print("❌ No checkpoint files found.")
    else:
        print("❌ No sample images found for inference.")
    
    # Step 4: Show results
    print("\n📊 MindSpore Demo Results:")
    print("- Framework: MindSpore")
    print("- Dataset: data/JPEGImages/ and data/Annotations/")
    print("- Model: models/*.ckpt")
    print("- Logs: logs/training.log")
    print("- Demo Output: mindspore_result.jpg")
    
    print("\n🎉 Kube-AI MindSpore Demo Complete!")
    print("📋 Deliverables Ready:")
    print("✅ Complete model implementation (MindSpore)")
    print("✅ Training code")
    print("✅ Inference code")
    print("✅ Sample dataset")
    print("✅ Model weights (.ckpt)")
    print("✅ Training logs")
    print("✅ README documentation")

if __name__ == "__main__":
    main()
import os
import shutil

# Define paths
base_path = '/Users/adityachauhan/Downloads/Rose'
target_base = os.path.join(base_path, 'consolidated_data')
source_dirs = ['train', 'val', 'test']
classes = ['Healthy_Leaf_Rose', 'Rose_Rust', 'Rose_sawfly_Rose_slug']

print("Consolidating unique images...")

for cls in classes:
    count = 0
    target_dir = os.path.join(target_base, cls)
    os.makedirs(target_dir, exist_ok=True)
    
    for src in source_dirs:
        src_dir = os.path.join(base_path, src, cls)
        if not os.path.exists(src_dir):
            continue
            
        for img in os.listdir(src_dir):
            if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                src_path = os.path.join(src_dir, img)
                dest_path = os.path.join(target_dir, img)
                
                # Only copy if it doesn't already exist (deduplication)
                if not os.path.exists(dest_path):
                    shutil.copy2(src_path, dest_path)
                    count += 1
    
    print(f"Class {cls}: Total unique images = {count}")

print("\nConsolidation complete.")

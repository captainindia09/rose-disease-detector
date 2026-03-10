import splitfolders
from ultralytics import YOLO
import os
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# 1. Dataset Splitting (70% Train, 15% Val, 15% Test)
input_folder = "/Users/adityachauhan/Downloads/Rose/consolidated_data"
output_folder = "/Users/adityachauhan/Downloads/Rose/dataset_v2"

print("Splitting dataset...")
# seed=42 for reproducibility
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(0.7, 0.15, 0.15))

# 2. Setup Training Environment
dataset_path = output_folder
model = YOLO('yolov8n-cls.pt')

print("\nStarting Clean Training...")
# 3. Train with Data Augmentation
# YOLOv8-cls has built-in augmentations (hsv, flip, etc.) enabled by default
results = model.train(
    data=dataset_path,
    epochs=10,        # Increased epochs for better generalization
    imgsz=224,
    project='rose_classification_v2',
    name='clean_train',
    optimizer='Adam',   # Robust optimizer
    # Augmentation parameters
    degrees=15,         # Rotate images
    flipud=0.5,         # Vertical flip
    fliplr=0.5,         # Horizontal flip
    shear=10            # Shear effect
)

# 4. Save the best model
best_model_path = os.path.join(results.save_dir, 'weights', 'best.pt')
root_best_path = "/Users/adityachauhan/Downloads/Rose/best_v2.pt"
shutil.copy2(best_model_path, root_best_path)
print(f"\nTraining Complete. Best model saved to: {root_best_path}")

# 5. Evaluation on Test Dataset
print("\nEvaluating on Test Dataset...")
# Load the best model cleanly
eval_model = YOLO(root_best_path)
test_results = eval_model.val(data=dataset_path, split='test')

# Extract and Print Metrics
print("\n--- FINAL TEST METRICS ---")
print(f"Top-1 Accuracy: {test_results.results_dict['metrics/accuracy_top1']:.4f}")

# Generate Confusion Matrix
# We can create a custom plot for easier reading
names = eval_model.names
y_true = []
y_pred = []

test_dir = os.path.join(dataset_path, 'test')
for class_folder in os.listdir(test_dir):
    class_idx = next(key for key, value in names.items() if value == class_folder)
    folder_path = os.path.join(test_dir, class_folder)
    
    # Run prediction on each image in the class folder
    results_list = eval_model(folder_path, verbose=False)
    for res in results_list:
        y_true.append(class_idx)
        y_pred.append(res.probs.top1)

# Plotting
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=names.values(), yticklabels=names.values(), cmap='Greens')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Rose Disease Classification')
plt.savefig('/Users/adityachauhan/Downloads/Rose/confusion_matrix_v2.png')
plt.show()

print("\nDetailed Classification Report:")
print(classification_report(y_true, y_pred, target_names=names.values()))

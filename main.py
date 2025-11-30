from ultralytics import YOLO
import cv2
import os
import numpy as np

model = YOLO('pin_detector.pt')

# Create output directory if it doesn't exist
output_dir = 'output_shrunk'
os.makedirs(output_dir, exist_ok=True)

# Run inference
results = model.predict('output', save=False, show_conf=False, show_labels=False)

# Shrink factor (0.9 means 90% of original size)
SHRINK_FACTOR = 0.9

for i, result in enumerate(results):
    # Get original image
    img = result.orig_img.copy()
    
    # Get bounding boxes
    boxes = result.boxes.xyxy.cpu().numpy()
    
    for box in boxes:
        x1, y1, x2, y2 = box
        
        # Calculate center and dimensions
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
        
        # Calculate new dimensions
        new_w = w * SHRINK_FACTOR
        new_h = h * SHRINK_FACTOR
        
        # Calculate new coordinates
        new_x1 = int(cx - new_w / 2)
        new_y1 = int(cy - new_h / 2)
        new_x2 = int(cx + new_w / 2)
        new_y2 = int(cy + new_h / 2)
        
        # Draw new box (Blue color, thickness 2)
        cv2.rectangle(img, (new_x1, new_y1), (new_x2, new_y2), (255, 0, 0), 2)
        
    # Save image
    # Use original filename if available, otherwise generate one
    if result.path:
        filename = os.path.basename(result.path)
    else:
        filename = f"result_{i}.jpg"
        
    save_path = os.path.join(output_dir, filename)
    cv2.imwrite(save_path, img)
    print(f"Saved {save_path}")
from ultralytics import YOLO
import cv2
import os
import numpy as np

def detect_pins(input_dir='output', output_dir='output_shrunk'):
    model = YOLO('pin_detector.pt')

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Run inference
    # Note: YOLO predict can take a directory path
    results = model.predict(input_dir, save=False, show_conf=False, show_labels=False)

    for i, result in enumerate(results):
        # Get original image
        img = result.orig_img.copy()
        
        # Get bounding boxes
        boxes = result.boxes.xyxy.cpu().numpy()
        
        for box in boxes:
            x1, y1, x2, y2 = box
            
            # Draw original box (Blue color matching detect_bad_pins.py target #052aff -> BGR: 255, 42, 5)
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 42, 5), 2)
            
        # Save image
        # Use original filename if available, otherwise generate one
        if result.path:
            filename = os.path.basename(result.path)
        else:
            filename = f"result_{i}.jpg"
            
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, img)
        print(f"Saved {save_path}")
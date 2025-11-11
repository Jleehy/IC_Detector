import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def main():
    # --- 1. SETUP ---
    
    # Define paths
    image_path = 'right.png'
    model_checkpoint = 'sam_vit_h_4b8939.pth' # The file you downloaded
    model_type = 'vit_h'
    
    # Set the device (use GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the original image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return

    # SAM expects images in RGB format, but OpenCV loads in BGR
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create an empty image to draw the outlines on
    outline_image = image.copy()

    print("Loading Segment Anything Model (SAM)...")
    # --- 2. LOAD THE ML MODEL ---
    
    sam = sam_model_registry[model_type](checkpoint=model_checkpoint)
    sam.to(device=device)

    # Use the Automatic Mask Generator
    mask_generator = SamAutomaticMaskGenerator(model=sam)

    print("Generating masks... (This might take a moment)")
    # --- 3. RUN THE ML MODEL ---
    
    masks = mask_generator.generate(image_rgb)
    
    print(f"Found {len(masks)} total objects (masks)")

    # --- 4. DRAW THE OUTLINES (WITH FILTERING) ---
    
    if len(masks) == 0:
        print("No masks found.")
        return

    # --- NEW: Define your area thresholds ---
    # !! You MUST tune these values for your image !!
    # This will ignore tiny masks (like the holes in the pins)
    min_area_threshold = 500  # Example value: 500 pixels
    # This will ignore huge masks (like the background)
    max_area_threshold = 50000 # Example value: 50,000 pixels

    masks_drawn = 0
    # Loop through each object (mask) the model found
    for i, mask_data in enumerate(masks):
        # Get the area (number of pixels) of the mask
        area = mask_data['area']
        
        # --- NEW: Uncomment this line to find your perfect thresholds ---
        # print(f"Mask {i} Area: {area}")

        # --- NEW: The filtering 'if' statement ---
        if min_area_threshold < area < max_area_threshold:
            # This mask is in our desired size range. Let's draw it.
            masks_drawn += 1
            
            mask = mask_data['segmentation']
            mask_binary = (mask * 255).astype(np.uint8)

            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw the contours on our 'outline_image'
            cv2.drawContours(outline_image, contours, -1, (0, 255, 0), 2)

    # Save the final image
    output_path = 'pins_outlined_filtered.png' # New output name
    cv2.imwrite(output_path, outline_image)
    
    print(f"Successfully drew {masks_drawn} filtered masks.")
    print(f"Saved filtered image to {output_path}")

if __name__ == "__main__":
    main()
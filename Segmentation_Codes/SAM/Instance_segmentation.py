import os
import cv2
import numpy as np
import random
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


def main():
    # --- Configuration ---
    model_type = "vit_h"  # Options: "vit_h", "vit_l", "vit_b"
    checkpoint_path = r"C:\Users\walat\segment-anything\sam_vit_h_4b8939.pth" # Update path

    # Directories
    input_dir = r"C:\Users\walat\OneDrive\Skrivebord\Mask2Former\Image\Instance\input"
    output_dir = r"C:\Users\walat\OneDrive\Skrivebord\Mask2Former\Image\Instance\output"
    os.makedirs(output_dir, exist_ok=True)

    # --- Load SAM Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to(device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    # --- Process Each Image ---
    jpg_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".jpg")]
    if not jpg_files:
        raise FileNotFoundError("No JPG files found in directory: " + input_dir)

    for filename in jpg_files:
        image_path = os.path.join(input_dir, filename)
        print("Processing:", image_path)

        # Read the image using OpenCV and convert to RGB
        image = cv2.imread(image_path)
        if image is None:
            print("Could not read image:", image_path)
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Generate instance masks automatically
        masks = mask_generator.generate(image_rgb)
        print(f"Found {len(masks)} masks for {filename}")

        # Create a copy for annotation
        annotated_image = image_rgb.copy()

        # For each mask, draw its contour and fill with a random color
        for mask in masks:
            segmentation = mask["segmentation"]
            color = np.array([random.randint(50, 255) for _ in range(3)], dtype=np.uint8)
            annotated_image[segmentation] = (0.5 * annotated_image[segmentation] + 0.5 * color).astype(np.uint8)

            # Draw bounding box if available
            if "bbox" in mask:
                x, y, w, h = mask["bbox"]
                cv2.rectangle(annotated_image, (int(x), int(y)), (int(x + w), int(y + h)), color.tolist(), 2)

        # Save output image
        out_filename = os.path.splitext(filename)[0] + "_SAM.png"
        out_path = os.path.join(output_dir, out_filename)
        cv2.imwrite(out_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        print("Saved output to:", out_path)


if __name__ == "__main__":
    main()

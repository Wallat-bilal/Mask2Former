import os
import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import numpy as np
import random
import cv2
import warnings

warnings.filterwarnings(
    "ignore", message="The following named arguments are not valid for `Mask2FormerImageProcessor.__init__`"
)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

def main():
    # --- Configuration ---
    # Pre-segmentation resolution: resize the input image before segmentation.
    pre_segmentation_resolution = (1280, 720)  # (width, height)
    # Post-segmentation resolution: resize the final output.
    post_segmentation_resolution = (1280, 720)  # (width, height)

    # --- Input & Output Directories ---
    input_dir = r"/home/wallat/Desktop/Mask2Former/Image/Semantic/input"
    output_dir = r"/home/wallat/Desktop/Mask2Former/Image/Semantic/output"
    os.makedirs(output_dir, exist_ok=True)

    jpg_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".jpg")]
    if not jpg_files:
        raise FileNotFoundError("No JPG files found in directory: " + input_dir)

    # --- Model & Processor Loading ---
    print("Starting semantic model download...")
    # Using a checkpoint fine-tuned for semantic segmentation (ADE20K)
    processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-ade-semantic", use_fast=False)
    model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-ade-semantic")
    print("Semantic model download complete!")

    # --- Helper Functions for Coloring ---
    def create_semantic_palette(segmentation):
        """Creates a color palette for each unique label in the segmentation map."""
        unique_labels = np.unique(segmentation)
        palette = {}
        for label in unique_labels:
            # Generate a random bright color (values between 50 and 255)
            palette[label] = [random.randint(50, 255) for _ in range(3)]
        return palette

    def apply_semantic_palette(segmentation, palette):
        """Maps each label in the segmentation to its color."""
        H, W = segmentation.shape
        colored = np.zeros((H, W, 3), dtype=np.uint8)
        for label, color in palette.items():
            colored[segmentation == label] = color
        return colored

    # --- Process Each Image ---
    for filename in jpg_files:
        input_image_path = os.path.join(input_dir, filename)
        print("Processing:", input_image_path)

        # Load the image using PIL and resize before segmentation
        image = Image.open(input_image_path)
        image = image.resize(pre_segmentation_resolution, Image.BILINEAR)

        # Preprocess the image
        inputs = processor(images=image, return_tensors="pt")

        # Run inference without gradients (runs on CPU by default)
        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process for semantic segmentation:
        # This returns a list of semantic segmentation maps (one per image)
        semantic_map = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0]
        # Convert the tensor to a NumPy array (segmentation map shape: [height, width])
        segmentation = semantic_map.cpu().numpy().astype(np.int32)

        # Create a color palette based on unique labels and apply it
        palette = create_semantic_palette(segmentation)
        colored_seg_map = apply_semantic_palette(segmentation, palette)

        # Resize the colored segmentation output to the target resolution
        resized_colored_seg_map = cv2.resize(colored_seg_map, post_segmentation_resolution, interpolation=cv2.INTER_LINEAR)

        # Save the output image
        output_filename = os.path.splitext(filename)[0] + "_semantic.png"
        output_file = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_file, cv2.cvtColor(resized_colored_seg_map, cv2.COLOR_RGB2BGR))
        print("Saved output to:", output_file)

    print("Processing complete!")

if __name__ == "__main__":
    main()

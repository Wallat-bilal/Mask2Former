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

def main():
    # --- Configuration ---
    fusion_labels = {3}  # Example: fuse label id 3; set to None if not needed.
    # Pre-segmentation resolution: resize the input image before passing it to the model.
    pre_segmentation_resolution = (1280, 720)  # (width, height)
    # Post-segmentation resolution: resize the final annotated output.
    post_segmentation_resolution = (1280, 720)  # (width, height)

    # --- Model & Processor Loading ---
    print("Starting model download...")
    processor = AutoImageProcessor.from_pretrained(
        "facebook/mask2former-swin-base-coco-panoptic", use_fast=False
    )
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-large-coco-panoptic"
    )
    print("Model download complete!")

    # --- Input & Output Directories ---
    input_dir = r"/home/wallat/Desktop/Mask2Former/Image/Panoptic/input"
    output_dir = r"/home/wallat/Desktop/Mask2Former/Image/Panoptic/output"
    os.makedirs(output_dir, exist_ok=True)

    jpg_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".jpg")]
    if not jpg_files:
        raise FileNotFoundError("No JPG files found in directory: " + input_dir)

    # --- Helper Functions ---
    def create_color_palette(segments_info):
        """Assigns a random bright color to each segment id."""
        palette = {}
        for seg in segments_info:
            seg_id = seg["id"]
            palette[seg_id] = [random.randint(50, 255) for _ in range(3)]
        return palette

    def apply_palette(segmentation, palette):
        """Creates an RGB image by mapping each segment id to its color."""
        H, W = segmentation.shape
        colored = np.zeros((H, W, 3), dtype=np.uint8)
        for seg_id, color in palette.items():
            colored[segmentation == seg_id] = color
        return colored

    # --- Process Each Image in the Input Directory ---
    for filename in jpg_files:
        input_image_path = os.path.join(input_dir, filename)
        print("Processing:", input_image_path)

        # Load the image using PIL
        image = Image.open(input_image_path)

        # Resize the image BEFORE segmentation
        image = image.resize(pre_segmentation_resolution, Image.BILINEAR)

        inputs = processor(images=image, return_tensors="pt")

        # Run inference (this code runs on CPU)
        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process for panoptic segmentation
        if fusion_labels is not None:
            result = processor.post_process_panoptic_segmentation(
                outputs,
                target_sizes=[image.size[::-1]],  # image.size is (width, height); reversed for (height, width)
                label_ids_to_fuse=fusion_labels
            )[0]
        else:
            result = processor.post_process_panoptic_segmentation(
                outputs,
                target_sizes=[image.size[::-1]]
            )[0]

        segmentation = result["segmentation"].cpu().numpy().astype(np.int32)
        segments_info = result["segments_info"]

        # Create a color palette and generate a colored segmentation map
        palette = create_color_palette(segments_info)
        colored_seg_map = apply_palette(segmentation, palette)

        # --- Annotate Segmentation with Labels ---
        annotated_img = colored_seg_map.copy()
        id2label = model.config.id2label if hasattr(model.config, "id2label") else {}

        for seg in segments_info:
            seg_id = seg["id"]
            label_id = seg["label_id"]
            if isinstance(id2label, dict):
                label_name = id2label.get(str(label_id), id2label.get(label_id, str(label_id)))
            else:
                label_name = str(label_id)

            ys, xs = np.where(segmentation == seg_id)
            if len(xs) == 0 or len(ys) == 0:
                continue
            center_x = int(np.mean(xs))
            center_y = int(np.mean(ys))
            cv2.putText(annotated_img, label_name, (center_x, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=2)

        # Resize the annotated image AFTER segmentation
        resized_annotated_img = cv2.resize(annotated_img, post_segmentation_resolution, interpolation=cv2.INTER_LINEAR)

        output_filename = os.path.splitext(filename)[0] + "_annotated.png"
        output_file = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_file, cv2.cvtColor(resized_annotated_img, cv2.COLOR_RGB2BGR))
        print("Saved output to:", output_file)

    print("Processing complete!")

if __name__ == "__main__":
    main()

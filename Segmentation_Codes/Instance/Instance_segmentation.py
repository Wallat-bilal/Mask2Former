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
    pre_segmentation_resolution = (1280, 720)  # (width, height)
    post_segmentation_resolution = (1280, 720)  # (width, height)

    # --- Model & Processor Loading ---
    print("Starting model download...")
    processor = AutoImageProcessor.from_pretrained(
        "facebook/mask2former-swin-base-coco-instance", use_fast=False
    )
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-large-coco-instance"
    )
    print("Model download complete!")

    # --- Input & Output Directories ---
    input_dir = r"C:\Users\walat\OneDrive\Skrivebord\Mask2Former\Image\Instance\input"
    output_dir = r"C:\Users\walat\OneDrive\Skrivebord\Mask2Former\Image\Instance\output"
    os.makedirs(output_dir, exist_ok=True)

    jpg_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".jpg")]
    if not jpg_files:
        raise FileNotFoundError("No JPG files found in directory: " + input_dir)

    # --- Helper Functions ---
    def create_color_palette(segments_info):
        """Assigns a random bright color to each segment id."""
        palette = {}
        for segment in segments_info:
            segment_id = segment["id"]
            palette[segment_id] = [random.randint(50, 255) for _ in range(3)]
        return palette

    def apply_palette(segmentation, palette):
        """Creates an RGB image by mapping each segment id to its color."""
        H, W = segmentation.shape
        colored = np.zeros((H, W, 3), dtype=np.uint8)
        for segment_id, color in palette.items():
            colored[segmentation == segment_id] = color
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

        # Post-process for instance segmentation
        result = processor.post_process_instance_segmentation(
            outputs,
            target_sizes=[image.size[::-1]]  # image.size is (width, height); reversed for (height, width)
        )[0]

        # Print result to inspect the structure
        print("Result keys:", result.keys())  # Print the keys of the result dictionary

        # Adjust based on the available keys in the result
        # Use 'segments_info' instead of 'instances_info'
        try:
            segmentation = result["segmentation"].cpu().numpy().astype(np.int32)
            segments_info = result["segments_info"]

            # Create a color palette and generate a colored segmentation map
            palette = create_color_palette(segments_info)
            colored_seg_map = apply_palette(segmentation, palette)

            # --- Annotate Segmentation with Labels ---
            annotated_img = colored_seg_map.copy()
            id2label = model.config.id2label if hasattr(model.config, "id2label") else {}

            for segment in segments_info:
                segment_id = segment["id"]
                label_id = segment["label_id"]
                if isinstance(id2label, dict):
                    label_name = id2label.get(str(label_id), id2label.get(label_id, str(label_id)))
                else:
                    label_name = str(label_id)

                ys, xs = np.where(segmentation == segment_id)
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

        except KeyError as e:
            print(f"KeyError: {e}. Check the result structure.")
            print(f"Available result: {result}")

    print("Processing complete!")

if __name__ == "__main__":
    main()

import sys
import argparse

# Add the directories containing your segmentation modules to the module search path.
segmentation_dir_1 = r"/home/wallat/Desktop/Mask2Former/Segmentation_Codes"
#segmentation_dir_2 = r"C:\Users\walat\OneDrive\Skrivebord\Mask2Former\Segmentation_Codes\SEEM"
#segmentation_dir_3 = r"C:\Users\walat\OneDrive\Skrivebord\Mask2Former\Segmentation_Codes\SAM"
segmentation_dir_2 = r"/home/wallat/Desktop/Mask2Former/Segmentation_Codes/Instance"

# Add both directories if they're not already in the system path.
if segmentation_dir_1 not in sys.path:
    sys.path.insert(0, segmentation_dir_1)
if segmentation_dir_2 not in sys.path:
    sys.path.insert(0, segmentation_dir_2)

def main():
    parser = argparse.ArgumentParser(
        description="Run a single segmentation mode: instance, panoptic, or semantic."
    )
    parser.add_argument(
        "--mode",
        choices=["instance", "panoptic", "semantic"],
        help="Segmentation mode to run: 'instance', 'panoptic', or 'semantic'."
    )
    args = parser.parse_args()

    mode = args.mode
    if mode is None:
        valid_modes = ["instance", "panoptic", "semantic"]
        while True:
            mode = input("Please choose one of the segmentation modes (instance, panoptic, semantic): ").strip().lower()
            if mode in valid_modes:
                break
            else:
                print("Invalid mode. Please choose one of: instance, panoptic, semantic.")

    if mode == "instance":
        print("Running Instance Segmentation...")
        try:
            # Update the import path based on the new segmentation directory
            from Instance import Instance_segmentation  # Correct import from the "Instance" directory
            Instance_segmentation.main()
        except ImportError as e:
            print(f"Error importing Instance Segmentation: {e}")
            sys.exit(1)

    elif mode == "panoptic":
        print("Running Panoptic Segmentation...")
        import Panoptic_segmentation
        Panoptic_segmentation.main()

    elif mode == "semantic":
        print("Running Semantic Segmentation...")
        import Semantic_segmentation
        Semantic_segmentation.main()

    else:
        print("Invalid mode selected. Please choose one of: instance, panoptic, semantic.")
        sys.exit(1)

if __name__ == "__main__":
    main()

#python main.py --mode semantic
#python main.py --mode instance
#python main.py --mode panoptic

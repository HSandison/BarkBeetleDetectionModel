# Install PyTorch for GPU
!pip install torch torchvision torchaudio

# Install Detectron2
!pip install -U torch torchvision
!pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Install OpenCV and pycocotools for handling images and COCO format
!pip install opencv-python-headless
!pip install pycocotools


# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')


# Import libraries
import os
import json
import random
import numpy as np
import cv2
from google.colab.patches import cv2_imshow

import torch
from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger
setup_logger()


# Define the new dataset directory
dataset_dir = "/content/drive/MyDrive/BarkBeetle/bark_beetle_detection_working"
images_dir = os.path.join(dataset_dir, 'images')

# Load the JSON annotation files
metadata_file = os.path.join(dataset_dir, 'annotations', 'COCO_IPS-IMAGE.json')
combined_annotations_file = os.path.join(dataset_dir, 'annotations', 'COCO_Combined-Images-for-Model.json')

with open(metadata_file, 'r') as f:
    metadata_annotations = json.load(f)

with open(combined_annotations_file, 'r') as f_combined:
    combined_annotations = json.load(f_combined)

# Ensure categories are consistent
combined_annotations['categories'] = metadata_annotations['categories']

# Save the combined annotations into a new JSON file
combined_annotations_path = os.path.join(dataset_dir, 'annotations', 'combined_annotations.json')
with open(combined_annotations_path, 'w') as f:
    json.dump(combined_annotations, f)

print(f"Combined annotations saved at {combined_annotations_path}")


# Create directories for train, val, and test within the dataset directory
train_dir = os.path.join(dataset_dir, 'train')
val_dir = os.path.join(dataset_dir, 'val')
test_dir = os.path.join(dataset_dir, 'test')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)


# Set seed for reproducibility
random.seed(42)

# Get image filenames and split the data
image_filenames = os.listdir(images_dir)
random.shuffle(image_filenames)

# Define split sizes
train_size = int(0.7 * len(image_filenames))  # 70% for training
val_size = int(0.15 * len(image_filenames))  # 15% for validation
test_size = len(image_filenames) - train_size - val_size  # Remaining 15% for testing

# Split filenames
train_filenames = image_filenames[:train_size]
val_filenames = image_filenames[train_size:train_size + val_size]
test_filenames = image_filenames[train_size + val_size:]

# Function to copy images to respective directories
def copy_images(filenames, source_dir, dest_dir):
    for filename in filenames:
        src = os.path.join(source_dir, filename)
        dst = os.path.join(dest_dir, filename)
        cv2.imwrite(dst, cv2.imread(src))

# Copy images to the respective directories
copy_images(train_filenames, images_dir, train_dir)
copy_images(val_filenames, images_dir, val_dir)
copy_images(test_filenames, images_dir, test_dir)

# Display the counts
print(f"Train images: {len(train_filenames)}, Val images: {len(val_filenames)}, Test images: {len(test_filenames)}")


def create_subset_annotations(filenames, annotations, subset_name):
    subset_annotations = {
        'images': [],
        'annotations': [],
        'categories': annotations['categories']
    }

    # Create a mapping for filename to image ID
    filename_to_id = {image['file_name']: image['id'] for image in annotations['images']}

    for filename in filenames:
        if filename in filename_to_id:
            image_id = filename_to_id[filename]
            # Add image info
            image_info = next(item for item in annotations['images'] if item['id'] == image_id)
            subset_annotations['images'].append(image_info)

            # Add corresponding annotations
            for ann in annotations['annotations']:
                if ann['image_id'] == image_id:
                    subset_annotations['annotations'].append(ann)

    return subset_annotations

# Create annotations for each subset
train_annotations = create_subset_annotations(train_filenames, combined_annotations, "train")
val_annotations = create_subset_annotations(val_filenames, combined_annotations, "val")
test_annotations = create_subset_annotations(test_filenames, combined_annotations, "test")

# Save filtered annotations in the annotations directory
with open(os.path.join(new_dataset_dir, 'annotations', "coco_annotations_train.json"), 'w') as f:
    json.dump(train_annotations, f)
with open(os.path.join(new_dataset_dir, 'annotations', "coco_annotations_val.json"), 'w') as f:
    json.dump(val_annotations, f)
with open(os.path.join(new_dataset_dir, 'annotations', "coco_annotations_test.json"), 'w') as f:
    json.dump(test_annotations, f)

print("Annotations created and saved for train, validation, and test sets.")



# Create annotations for each subset
train_annotations = create_subset_annotations(train_filenames, combined_annotations, "train")
val_annotations = create_subset_annotations(val_filenames, combined_annotations, "val")
test_annotations = create_subset_annotations(test_filenames, combined_annotations, "test")

# Save filtered annotations in the annotations directory
with open(os.path.join(dataset_dir, 'annotations', "coco_annotations_train.json"), 'w') as f:
    json.dump(train_annotations, f)
with open(os.path.join(dataset_dir, 'annotations', "coco_annotations_val.json"), 'w') as f:
    json.dump(val_annotations, f)
with open(os.path.join(dataset_dir, 'annotations', "coco_annotations_test.json"), 'w') as f:
    json.dump(test_annotations, f)

print("Annotations created and saved for train, validation, and test sets.")


# Check for duplicate annotation IDs in combined_annotations
def check_duplicates(annotations):
    ann_ids = [ann['id'] for ann in annotations['annotations']]
    unique_ids = set(ann_ids)
    if len(unique_ids) != len(ann_ids):
        print(f"Warning: Duplicate annotation IDs found: {len(ann_ids) - len(unique_ids)} duplicates.")
    else:
        print("No duplicate annotation IDs found in the combined annotations.")

# Call the function to check for duplicates
check_duplicates(combined_annotations)



def find_duplicate_ids(annotations):
    ann_ids = [ann['id'] for ann in annotations['annotations']]
    id_count = {}

    # Count occurrences of each ID
    for id in ann_ids:
        if id in id_count:
            id_count[id] += 1
        else:
            id_count[id] = 1

    # Find duplicates
    duplicates = {id: count for id, count in id_count.items() if count > 1}

    if duplicates:
        print(f"Duplicate annotation IDs found: {duplicates}")
    else:
        print("No duplicates found.")

# Call the function to find duplicates
find_duplicate_ids(combined_annotations)


from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

# Register the datasets
register_coco_instances("bark_beetle_train", {}, os.path.join(dataset_dir, 'annotations', "coco_annotations_train.json"), train_dir)
register_coco_instances("bark_beetle_val", {}, os.path.join(dataset_dir, 'annotations', "coco_annotations_val.json"), val_dir)
register_coco_instances("bark_beetle_test", {}, os.path.join(dataset_dir, 'annotations', "coco_annotations_test.json"), test_dir)

# Verify registration
train_metadata = MetadataCatalog.get("bark_beetle_train")
val_metadata = MetadataCatalog.get("bark_beetle_val")

print("Registered training dataset with", len(DatasetCatalog.get("bark_beetle_train")), "images.")
print("Registered validation dataset with", len(DatasetCatalog.get("bark_beetle_val")), "images.")


# Configure the model for training
from detectron2.config import get_cfg

# Initialize configuration
cfg = get_cfg()

# Load the Mask R-CNN model configuration
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

# Set the dataset names
cfg.DATASETS.TRAIN = ("bark_beetle_train",)
cfg.DATASETS.TEST = ("bark_beetle_val",)  # Validation dataset

# Data loader settings
cfg.DATALOADER.NUM_WORKERS = 2  # Number of workers for data loading

# Solver settings
cfg.SOLVER.IMS_PER_BATCH = 2  # Batch size
cfg.SOLVER.BASE_LR = 0.00025  # Learning rate
cfg.SOLVER.MAX_ITER = 2 #3000  # Number of iterations

# Checkpoint settings
cfg.SOLVER.CHECKPOINT_PERIOD = 500  # Save checkpoint every 500 iterations

# Output directory
cfg.MODEL.OUTPUT_DIR = "./output"  # Directory to save model outputs

# Set the number of classes in the model (including the background)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(train_metadata.get("thing_classes", []))

# Create output directory
os.makedirs(cfg.MODEL.OUTPUT_DIR, exist_ok=True)


# Check for GPU availability
import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("GPU is available!")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("GPU is not available.")


import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

print("Detectron2 is installed correctly!")


# Create trainer and start training
from detectron2.engine import DefaultTrainer

# Create a trainer
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)  # Start training from scratch
trainer.train()  # Start the training



# Check model has saved in Google Colab
import os

# Specify the output directory
output_dir = "./output"
if os.path.exists(output_dir):
    print(f"Model saved in: {output_dir}")
    print("Contents:", os.listdir(output_dir))
else:
    print("Output directory does not exist!")

# Save the model in Google Drive
torch.save(trainer.model.state_dict(), '/content/drive/MyDrive/BarkBeetle/bark_beetle_detection_working/model/model_final.pth')



# Validation Code
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import inference_on_dataset

# Create COCO Evaluator
evaluator = COCOEvaluator("bark_beetle_val", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "bark_beetle_val")

# Evaluate the model on the validation set
inference_on_dataset(trainer.model, val_loader, evaluator)


# Import necessary libraries
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.checkpoint import DetectionCheckpointer  # Import DetectionCheckpointer

# Initialize configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

# Specify the correct number of classes
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5  # Change this to your actual number of classes

# # Load the weights from the trained model
# cfg.MODEL.WEIGHTS = './output/model_final.pth'  # Adjust the path if needed
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set the threshold for this model

# Load the weights from Google Drive
cfg.MODEL.WEIGHTS = '/content/drive/MyDrive/BarkBeetle/bark-beetle-detection_working/model/model_final.pth'

# Force the code to run on CPU
cfg.MODEL.DEVICE = "cpu"

# Create predictor
predictor = DefaultPredictor(cfg)

# Create checkpointer and load weights (without weights_only)
checkpointer = DetectionCheckpointer(predictor.model)
checkpointer.load(cfg.MODEL.WEIGHTS)  # Removed weights_only parameter



import cv2
import random
import os
import matplotlib.pyplot as plt
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Correct image folder path
image_folder_path = "/content/drive/MyDrive/BarkBeetle/bark_beetle_detection_working/test/"
predictions_folder_path = "/content/drive/MyDrive/BarkBeetle/bark_beetle_detection_working/predictions/"

# Create predictions folder if it doesn't exist
os.makedirs(predictions_folder_path, exist_ok=True)

# List images in the folder to ensure it contains the expected files
print("Listing images in the folder:")
image_files = os.listdir(image_folder_path)  # List images in the folder
print(image_files)

# Load a random image from the directory
#sample_image_path = os.path.join(image_folder_path, random.choice(image_files))  # Choose a random image
sample_image_path = os.path.join(image_folder_path, ('WE203_26_2.jpg'))
image = cv2.imread(sample_image_path)

# Check if the image was loaded correctly
if image is None:
    print("Error loading the image.")
else:
    print(f"Loaded image: {sample_image_path}")  # Print loaded image path

    # Make predictions
    outputs = predictor(image)

    # Check predictions
    print("Predictions:", outputs)  # Inspect the outputs
    if outputs["instances"].has("pred_classes"):
        print("Number of instances predicted:", len(outputs["instances"]))
    else:
        print("No instances predicted.")

    # Visualize the predictions
    metadata = MetadataCatalog.get("bark_beetle_val")
    v = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.0)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # Create a new filename with '_prediction' suffix
    base_filename = os.path.basename(sample_image_path)  # Get original filename
    new_filename = os.path.splitext(base_filename)[0] + "_prediction.jpg"  # Append '_prediction' to the name
    saved_image_path = os.path.join(predictions_folder_path, new_filename)  # Create new file path

    # Save the visualized image with predictions
    cv2.imwrite(saved_image_path, v.get_image()[:, :, ::-1])  # Save the image in BGR format

    print(f"Saved image with predictions to: {saved_image_path}")

    # Create directories for each predicted class
    instances = outputs["instances"].to("cpu")
    classes = instances.pred_classes.numpy()  # Get predicted classes
    boxes = instances.pred_boxes.tensor.numpy()  # Get bounding boxes

    # Get the unique classes
    unique_classes = set(classes)
    class_folders = {}

    # Create a folder for each class
    for cls in unique_classes:
        class_name = metadata.thing_classes[cls]  # Get class name
        class_folder_path = os.path.join(predictions_folder_path, class_name)
        os.makedirs(class_folder_path, exist_ok=True)
        class_folders[class_name] = class_folder_path  # Store the class folder path

    # Extract and save each prediction
    for i in range(len(classes)):
        class_name = metadata.thing_classes[classes[i]]
        box = boxes[i].astype(int)  # Convert to integer for indexing

        # Extract the bounding box
        x1, y1, x2, y2 = box
        extracted_image = image[y1:y2, x1:x2]

        # Create a unique filename for each extracted object
        extracted_filename = os.path.join(class_folders[class_name], f"{base_filename}_class_{class_name}_{i}.jpg")
        cv2.imwrite(extracted_filename, extracted_image)  # Save the extracted image

        print(f"Saved extracted image to: {extracted_filename}")

    # Show the image with predictions using matplotlib
    plt.figure(figsize=(12, 8))
    plt.imshow(v.get_image()[:, :, ::-1])  # Convert BGR to RGB
    plt.axis('off')  # Hide axes
    plt.show()





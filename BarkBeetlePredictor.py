# Import necessary libraries
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
import cv2
import os
import matplotlib.pyplot as plt
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

# Define dataset directory and paths
dataset_dir = "/content/drive/MyDrive/BarkBeetle/bark_beetle_detection_working"
train_dir = os.path.join(dataset_dir, 'train')
val_dir = os.path.join(dataset_dir, 'val')
test_dir = os.path.join(dataset_dir, 'test')

# Register the datasets
register_coco_instances("bark_beetle_train", {}, os.path.join(dataset_dir, 'annotations', "coco_annotations_train.json"), train_dir)
register_coco_instances("bark_beetle_val", {}, os.path.join(dataset_dir, 'annotations', "coco_annotations_val.json"), val_dir)
register_coco_instances("bark_beetle_test", {}, os.path.join(dataset_dir, 'annotations', "coco_annotations_test.json"), test_dir)

# Verify registration
train_metadata = MetadataCatalog.get("bark_beetle_train")
val_metadata = MetadataCatalog.get("bark_beetle_val")
print("Registered training dataset with", len(DatasetCatalog.get("bark_beetle_train")), "images.")
print("Registered validation dataset with", len(DatasetCatalog.get("bark_beetle_val")), "images.")

# Initialize configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

# Specify the number of classes (adjust this to match your trained model)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5  # Change this to your actual number of classes

# Load the weights from the pre-trained model
cfg.MODEL.WEIGHTS = '/content/drive/MyDrive/BarkBeetle/bark_beetle_detection_working/model/model_final.pth'

# Force the code to run on CPU (or use "cuda" if you want to utilize GPU)
cfg.MODEL.DEVICE = "cpu"

# Create predictor
predictor = DefaultPredictor(cfg)

# Path to the test images folder
image_folder_path = test_dir  # Using the test directory defined earlier
predictions_folder_path = os.path.join(dataset_dir, 'predictions/')

# Create predictions folder if it doesn't exist
os.makedirs(predictions_folder_path, exist_ok=True)

# List images in the folder to ensure it contains the expected files
print("Listing images in the folder:")
image_files = os.listdir(image_folder_path)  # List images in the folder
print(image_files)

# Iterate over images in the test folder and make predictions
for image_file in image_files:
    sample_image_path = os.path.join(image_folder_path, image_file)

    # Load each image one at a time
    image = cv2.imread(sample_image_path)

    # Check if the image was loaded correctly
    if image is None:
        print(f"Error loading the image: {sample_image_path}")
        continue

    # Make predictions
    outputs = predictor(image)

    # Check predictions
    print("Predictions:", outputs)  # Inspect the outputs
    if outputs["instances"].has("pred_classes"):
        instances = outputs["instances"]
        instances = instances[instances.scores > 0.4]  # Set a threshold for score to reduce instances
        print("Number of instances predicted:", len(instances))
    else:
        print("No instances predicted.")

    # Visualize the predictions with class names and confidence scores
    metadata = MetadataCatalog.get("bark_beetle_val")  # Get metadata for visualization
    v = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.0)

    # Add instance predictions to the visualizer
    v = v.draw_instance_predictions(instances.to("cpu"))

    # Create a new filename with '_prediction' suffix
    new_filename = os.path.splitext(image_file)[0] + "_prediction.jpg"  # Append '_prediction' to the name
    saved_image_path = os.path.join(predictions_folder_path, new_filename)  # Create new file path

    # Save the visualized image with predictions
    cv2.imwrite(saved_image_path, v.get_image()[:, :, ::-1])  # Save the image in BGR format
    print(f"Saved image with predictions to: {saved_image_path}")

    # Show the image with predictions using matplotlib
    plt.figure(figsize=(12, 8))
    plt.imshow(v.get_image()[:, :, ::-1])  # Convert BGR to RGB for display
    plt.axis('off')  # Hide axes
    plt.show()  # Display the image

    # Clear variables to free memory
    del image, outputs, instances, v

# Optionally, add a message when all images have been processed
print("All images processed successfully.")


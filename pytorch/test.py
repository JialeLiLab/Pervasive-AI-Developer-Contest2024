import torch
from torch import nn
from resnet18 import Resnet18  # Ensure Resnet18 is correctly imported
from PIL import Image
from torchvision import transforms
import os
import time
import pynvml
import warnings

warnings.filterwarnings('always')

def load_images_from_folder(folder):
    """
    Load images and their labels from the given folder.

    Args:
        folder (str): Path to the folder containing images.

    Returns:
        images (list): List of image file paths.
        labels (list): List of labels corresponding to the images.
    """
    images = []
    labels = []
    for label in ['Defective', 'Non defective']:  # Assuming labels are 'Defective' and 'Non defective'
        class_folder = os.path.join(folder, label)
        for filename in os.listdir(class_folder):
            if filename.endswith(".jpg"):
                img_path = os.path.join(class_folder, filename)
                images.append(img_path)
                labels.append(label)
    return images, labels

# Initialize pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming using the first GPU

# Set device
device = "cuda" if torch.cuda.is_available() else 'cpu'

# Load and prepare model
model = Resnet18(num_classes=2).to(device)  # Set number of classes to 2
model.load_state_dict(torch.load("./save_model/m6_resnet18_epoch20_trainacc0.7699.pth"))
model.eval()

# Data preprocessing
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4315, 0.3989, 0.3650], [0.2250, 0.2176, 0.2111])  # Normalization
])

# Load test dataset
folder_path = "./128_128dataset/Test"  # Update to your folder path
images, labels = load_images_from_folder(folder_path)

total_predictions, correct_predictions = 0, 0
total_inference_time = 0.0
total_power_consumption = 0.0

# Perform predictions
for img_path, true_label in zip(images, labels):
    img = Image.open(img_path).convert('RGB')
    img = data_transform(img)
    img = img.unsqueeze(0).to(device)

    start_time = time.time()  # Start timing
    power_start = pynvml.nvmlDeviceGetPowerUsage(handle)  # Start power usage

    with torch.no_grad():
        outputs = model(img)
        probabilities = nn.functional.softmax(outputs, dim=1)
        predicted = torch.argmax(probabilities, 1)
        predicted_label = 'Defective' if predicted == 0 else 'Non defective'
        predicted_probability = probabilities[0, predicted].item()

    power_end = pynvml.nvmlDeviceGetPowerUsage(handle)  # End power usage
    inference_time = time.time() - start_time  # End timing

    # Calculate power consumption difference
    power_consumption = (power_end - power_start) / 1000  # Convert to watts
    total_power_consumption += power_consumption

    # Accumulate statistics
    correct_predictions += (predicted_label == true_label)
    total_predictions += 1
    total_inference_time += inference_time

    print(f"Image: {os.path.basename(img_path)}, True label: {true_label}, Predicted label: {predicted_label}, Probability: {predicted_probability:.10f}, Inference Time: {inference_time:.10f} seconds, Power Consumption: {power_consumption:.4f} W")

# Output overall prediction accuracy and inference time
accuracy = correct_predictions / total_predictions
average_inference_time = total_inference_time / total_predictions
average_power_consumption = total_power_consumption / total_predictions

print(f'Overall Accuracy: {accuracy:.3f}')
print(f'Total Inference Time: {total_inference_time:.10f} seconds')
print(f'Average Inference Time per Image: {average_inference_time:.10f} seconds')
print(f'Total Power Consumption: {total_power_consumption:.10f} W')
print(f'Average Power Consumption per Image: {average_power_consumption:.10f} W')

# Shutdown pynvml
pynvml.nvmlShutdown()

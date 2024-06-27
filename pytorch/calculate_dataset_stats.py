from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

# Ensure MyDataSet is imported from the corresponding file
from my_dataset import MyDataSet

def read_data(data_path):
    import os
    images_path = []
    images_class = []
    for class_dir in os.listdir(data_path):
        class_dir_path = os.path.join(data_path, class_dir)
        for img_file in os.listdir(class_dir_path):
            images_path.append(os.path.join(class_dir_path, img_file))
            images_class.append(class_dir)
    return images_path, images_class

# Data transformation to convert images to tensors
data_transform = transforms.Compose([
    transforms.ToTensor()
])

# Read data
train_images_path, train_images_label = read_data('./128_128dataset_Improved/128_128dataset_Improved/val')

# Create dataset
train_dataset = MyDataSet(images_path=train_images_path,
                          images_class=train_images_label,
                          transform=data_transform)

# Create data loader
train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=False, num_workers=0)

def compute_mean_std(dataloader):
    mean = 0.
    std = 0.
    total_images_count = 0

    for images, _ in tqdm(dataloader, desc="Computing Dataset Mean and Std"):
        # Number of images in batch
        batch_samples = images.size(0)
        # Reshape to [batch_size, channels, number_of_pixels]
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    # Compute overall mean and std
    mean /= total_images_count
    std /= total_images_count

    return mean, std

# Compute mean and standard deviation
mean, std = compute_mean_std(train_dataloader)
print(f'Mean: {mean}')
print(f'Std: {std}')

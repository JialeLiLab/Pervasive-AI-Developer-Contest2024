import torch
from torch import nn
from resnet18 import Resnet18  # Ensure the Resnet18 class is correctly imported
from torch.optim import lr_scheduler
from torchvision import transforms
import os
from tqdm import tqdm
from my_dataset import MyDataSet  # Import custom dataset class

# Update data preprocessing with various transformations for data augmentation
data_transform = {
    "train": transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.RandomRotation(10),  # Randomly rotate the image by up to 10 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Randomly change the brightness, contrast, and saturation
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),  # Apply random affine transformations with shear and scale
        transforms.RandomCrop(128, padding=4),  # Randomly crop the image with padding
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize([0.4315, 0.3989, 0.3650], [0.2250, 0.2176, 0.2111])  # Normalize the image with mean and standard deviation
    ]),
    "val": transforms.Compose([
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize([0.4315, 0.3989, 0.3650], [0.2250, 0.2176, 0.2111])  # Normalize the image with mean and standard deviation
    ])
}

def read_data(data_path):
    """
    Read image paths and labels from the dataset directory.

    Args:
        data_path (str): Path to the dataset directory.

    Returns:
        images_path (list): List of image file paths.
        images_label (list): List of image labels corresponding to the file paths.
    """
    images_path = []
    images_label = []
    for class_dir in os.listdir(data_path):
        class_dir_path = os.path.join(data_path, class_dir)
        for img_file in os.listdir(class_dir_path):
            images_path.append(os.path.join(class_dir_path, img_file))
            images_label.append(class_dir)
    return images_path, images_label

if __name__ == '__main__':
    # Load train and validation data
    train_images_path, train_images_label = read_data('./128_128dataset_Improved/128_128dataset_Improved/train')
    val_images_path, val_images_label = read_data('./128_128dataset_Improved/128_128dataset_Improved/val')

    # Create dataset objects for training and validation
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    from torch.utils.data import DataLoader

    # Create data loaders for training and validation datasets
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    test_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    device = "cuda" if torch.cuda.is_available() else 'cpu'  # Use GPU if available
    model = Resnet18(num_classes=2).to(device)  # Initialize the Resnet18 model
    loss_fn = nn.CrossEntropyLoss()  # Define the loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)  # Add L2 regularization
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Learning rate scheduler

    def train(dataloader, model, loss_fn, optimizer, epoch):
        """
        Train the model for one epoch.

        Args:
            dataloader (DataLoader): DataLoader for training data.
            model (nn.Module): The model to train.
            loss_fn (nn.Module): Loss function.
            optimizer (torch.optim.Optimizer): Optimizer.
            epoch (int): Current epoch number.

        Returns:
            average_acc (float): Average training accuracy.
        """
        model.train()
        loop = tqdm(dataloader, desc=f'\033[97m[Train epoch {epoch}]', total=len(dataloader), leave=True)
        total_acc = 0
        total_loss = 0
        total_samples = 0

        for X, y in loop:
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = loss_fn(output, y)
            _, predicted = torch.max(output.data, 1)
            acc = (predicted == y).sum().item()
            total_acc += acc
            total_samples += y.size(0)
            total_loss += loss.item() * y.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            average_loss = total_loss / total_samples
            average_acc = total_acc / total_samples
            loop.set_postfix(loss=average_loss, acc=average_acc)

        return average_acc

    def val(dataloader, model, loss_fn, epoch):
        """
        Validate the model for one epoch.

        Args:
            dataloader (DataLoader): DataLoader for validation data.
            model (nn.Module): The model to validate.
            loss_fn (nn.Module): Loss function.
            epoch (int): Current epoch number.

        Returns:
            average_acc (float): Average validation accuracy.
        """
        model.eval()
        loop = tqdm(dataloader, desc=f'\033[97m[Valid epoch {epoch}]', total=len(dataloader), leave=True)
        total_loss, total_acc, n = 0.0, 0.0, 0
        with torch.no_grad():
            for X, y in loop:
                X, y = X.to(device), y.to(device)
                output = model(X)
                loss = loss_fn(output, y)
                _, predicted = torch.max(output.data, 1)
                acc = (predicted == y).sum().item() / y.size(0)

                total_loss += loss.item()
                total_acc += acc
                n += 1

                loop.set_postfix(loss=total_loss / n, acc=total_acc / n)

        return total_acc / n

    epochs = 100
    best_train_accuracy = 0.0
    best_val_accuracy = 0.0
    for epoch in range(epochs):
        train_accuracy = train(train_dataloader, model, loss_fn, optimizer, epoch)
        val_accuracy = val(test_dataloader, model, loss_fn, epoch)
        lr_scheduler.step()

        model_saved = False  # Flag to check if model was saved in this epoch

        # Save the model with the best validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            os.makedirs('save_model', exist_ok=True)
            torch.save(model.state_dict(), f'save_model/m8_resnet18_epoch{epoch}_valacc{val_accuracy:.4f}.pth')
            tqdm.write(f'Saving best val model')
            model_saved = True

        # Save the model with the best training accuracy if validation model was not saved
        if train_accuracy > best_train_accuracy and not model_saved:
            best_train_accuracy = train_accuracy
            os.makedirs('save_model', exist_ok=True)
            torch.save(model.state_dict(), f'save_model/m7_resnet18_trainacc{train_accuracy:.4f}.pth')
            tqdm.write(f'Saving best train model')

    print('Training complete.')

from PIL import Image
import torch
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    """Custom Dataset for loading images and their corresponding labels."""

    def __init__(self, images_path: list, images_class: list, transform=None):
        """
        Args:
            images_path (list): List of paths to the images.
            images_class (list): List of class labels corresponding to the images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform
        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(set(images_class)))}

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.images_path)

    def __getitem__(self, item):
        """
        Args:
            item (int): Index of the sample to be fetched.

        Returns:
            tuple: (image, label) where image is the transformed image and label is the corresponding class index.
        """
        img = Image.open(self.images_path[item])
        if img.mode != 'RGB':
            img = img.convert('RGB')

        label = self.class_to_idx[self.images_class[item]]  # Convert label using the mapping

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function to combine a list of samples into a batch.

        Args:
            batch (list): List of tuples where each tuple is (image, label).

        Returns:
            tuple: (images, labels) where images is a tensor of images and labels is a tensor of labels.
        """
        # Reference for default_collate:
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = zip(*batch)

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels

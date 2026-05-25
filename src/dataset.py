import numpy as np
import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


# clipLimit=2.0 caps amplification to avoid boosting noise on uniform regions;
# tileGridSize=(8,8) produces 64x64px tiles on 512x512 input, standard for chest X-rays.
def apply_clahe(image):
    if isinstance(image, Image.Image):
        image = np.array(image, dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return Image.fromarray(clahe.apply(image))


def get_transforms(img_size):
    # CLAHE runs before ColorJitter so it enhances the clean, unperturbed image.
    # The original order (ColorJitter → CLAHE) was wrong: CLAHE re-normalised
    # local contrast and partially cancelled whatever variation ColorJitter added.
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Lambda(apply_clahe),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Lambda(apply_clahe),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])

    return train_transform, val_transform


class ChestXrayDataset(Dataset):
    def __init__(self, dataframe, diseases, transform=None,
                 return_path=False, return_image_id=False):
        if return_path and return_image_id:
            raise ValueError(
                "Pass only one of return_path or return_image_id, not both"
            )
        self.df = dataframe.reset_index(drop=True)
        self.diseases = diseases
        self.transform = transform
        self.return_path = return_path
        self.return_image_id = return_image_id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['image_path']
        image = Image.open(image_path).convert('L')

        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(
            [row[disease] for disease in self.diseases],
            dtype=torch.float32
        )

        if self.return_path:
            return image, labels, image_path
        if self.return_image_id:
            return image, labels, row['Image Index']
        return image, labels

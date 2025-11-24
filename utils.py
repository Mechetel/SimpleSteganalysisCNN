import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class SteganalysisDataset(Dataset):
    """
    Steganalysis dataset for binary classification
    Label 0: Cover images (clean)
    Label 1: Stego images (with hidden data)
    """
    def __init__(self, cover_path, stego_path, transform=None):
        self.cover_path = os.path.expanduser(cover_path)
        self.stego_path = os.path.expanduser(stego_path)
        self.transform = transform
        self.images = []
        self.labels = []

        # Load cover images (label 0)
        if os.path.exists(self.cover_path):
            cover_files = [f for f in os.listdir(self.cover_path)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.pgm'))]
            for img_name in cover_files:
                self.images.append(os.path.join(self.cover_path, img_name))
                self.labels.append(0)
            print(f"Loaded {len(cover_files)} cover images from {self.cover_path}")
        else:
            print(f"Warning: Cover path not found: {self.cover_path}")

        # Load stego images (label 1)
        if os.path.exists(self.stego_path):
            stego_files = [f for f in os.listdir(self.stego_path)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.pgm'))]
            for img_name in stego_files:
                self.images.append(os.path.join(self.stego_path, img_name))
                self.labels.append(1)
            print(f"Loaded {len(stego_files)} stego images from {self.stego_path}")
        else:
            print(f"Warning: Stego path not found: {self.stego_path}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]

        # Handle grayscale images (common in steganalysis)
        image = Image.open(img_path)
        if image.mode == 'L':  # Grayscale
            image = image.convert('RGB')  # Convert to RGB for consistency
        elif image.mode != 'RGB':
            image = image.convert('RGB')

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)


def get_transforms(img_size=256, augment=True):
    """
    Get data transforms for training and validation
    Note: For steganalysis, aggressive augmentation should be avoided
    as it might affect the hidden data patterns
    """
    if augment:
        # Minimal augmentation for steganalysis
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def get_data_loaders(cover_path, stego_path, valid_cover_path=None,
                     valid_stego_path=None, batch_size=32, img_size=256, augment=True):
    """Create data loaders for training and validation"""
    train_transform, val_transform = get_transforms(img_size, augment)

    # Training dataset
    train_dataset = SteganalysisDataset(cover_path, stego_path, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=4, pin_memory=True)

    # Validation dataset (if paths provided)
    val_loader = None
    if valid_cover_path and valid_stego_path:
        val_dataset = SteganalysisDataset(valid_cover_path, valid_stego_path,
                                         transform=val_transform)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader


def get_test_loader(cover_path, stego_path, batch_size=32, img_size=256):
    """Create data loader for testing"""
    _, test_transform = get_transforms(img_size, augment=False)

    test_dataset = SteganalysisDataset(cover_path, stego_path, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=4, pin_memory=True)

    return test_loader


def calculate_metrics(y_true, y_pred):
    """Calculate classification metrics"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(model, optimizer, filepath, device):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from {filepath} (Epoch {epoch}, Loss: {loss:.4f})")
    return epoch, loss


# ==================== SRM Filter Initialization ====================

def get_srm_kernels():
    """
    Initialize SRM (Spatial Rich Model) high-pass filter kernels
    Using the exact filters from the provided configuration
    """
    # Filter Class 1 (8 filters) - First-order derivatives
    filter_class_1 = [
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=np.float32),
        np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]], dtype=np.float32),
        np.array([[0, 0, 1], [0, -1, 0], [0, 0, 0]], dtype=np.float32),
        np.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]], dtype=np.float32),
        np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype=np.float32),
        np.array([[0, 0, 0], [0, -1, 0], [1, 0, 0]], dtype=np.float32),
        np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]], dtype=np.float32),
        np.array([[0, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32)
    ]

    # Filter Class 2 (4 filters) - Second-order derivatives (normalized by 2)
    filter_class_2 = [
        np.array([[1, 0, 0], [0, -2, 0], [0, 0, 1]], dtype=np.float32) / 2,
        np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]], dtype=np.float32) / 2,
        np.array([[0, 0, 1], [0, -2, 0], [1, 0, 0]], dtype=np.float32) / 2,
        np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]], dtype=np.float32) / 2,
    ]

    # Filter Class 3 (8 filters) - Third-order derivatives (normalized by 3)
    filter_class_3 = [
        np.array([[-1, 0, 0, 0, 0], [0, 3, 0, 0, 0], [0, 0, -3, 0, 0],
                  [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]], dtype=np.float32) / 3,
        np.array([[0, 0, -1, 0, 0], [0, 0, 3, 0, 0], [0, 0, -3, 0, 0],
                  [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]], dtype=np.float32) / 3,
        np.array([[0, 0, 0, 0, -1], [0, 0, 0, 3, 0], [0, 0, -3, 0, 0],
                  [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=np.float32) / 3,
        np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -3, 3, -1],
                  [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=np.float32) / 3,
        np.array([[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, -3, 0, 0],
                  [0, 0, 0, 3, 0], [0, 0, 0, 0, -1]], dtype=np.float32) / 3,
        np.array([[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, -3, 0, 0],
                  [0, 0, 3, 0, 0], [0, 0, -1, 0, 0]], dtype=np.float32) / 3,
        np.array([[0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, -3, 0, 0],
                  [0, 3, 0, 0, 0], [-1, 0, 0, 0, 0]], dtype=np.float32) / 3,
        np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [-1, 3, -3, 1, 0],
                  [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=np.float32) / 3
    ]

    # Edge filters 3x3 (4 filters) - normalized by 4
    filter_edge_3x3 = [
        np.array([[-1, 2, -1], [2, -4, 2], [0, 0, 0]], dtype=np.float32) / 4,
        np.array([[0, 2, -1], [0, -4, 2], [0, 2, -1]], dtype=np.float32) / 4,
        np.array([[0, 0, 0], [2, -4, 2], [-1, 2, -1]], dtype=np.float32) / 4,
        np.array([[-1, 2, 0], [2, -4, 0], [-1, 2, 0]], dtype=np.float32) / 4,
    ]

    # Square 3x3 (1 filter) - normalized by 4
    square_3x3 = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32) / 4

    # Edge filters 5x5 (4 filters) - normalized by 12
    filter_edge_5x5 = [
        np.array([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2],
                  [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=np.float32) / 12,
        np.array([[0, 0, -2, 2, -1], [0, 0, 8, -6, 2], [0, 0, -12, 8, -2],
                  [0, 0, 8, -6, 2], [0, 0, -2, 2, -1]], dtype=np.float32) / 12,
        np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [-2, 8, -12, 8, -2],
                  [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]], dtype=np.float32) / 12,
        np.array([[-1, 2, -2, 0, 0], [2, -6, 8, 0, 0], [-2, 8, -12, 0, 0],
                  [2, -6, 8, 0, 0], [-1, 2, -2, 0, 0]], dtype=np.float32) / 12,
    ]

    # Square 5x5 (1 filter) - normalized by 12
    square_5x5 = np.array([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2],
                           [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]], dtype=np.float32) / 12

    # Combine all filters (total: 8 + 4 + 8 + 4 + 1 + 4 + 1 = 30 filters)
    all_filters = filter_class_1 + filter_class_2 + filter_class_3 + \
                  filter_edge_3x3 + [square_3x3] + filter_edge_5x5 + [square_5x5]

    return all_filters

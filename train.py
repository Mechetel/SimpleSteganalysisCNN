import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import argparse
from pathlib import Path
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model import StegoCNN

# Custom Dataset for Steganography Detection
class StegoDataset(Dataset):
    def __init__(self, cover_path, stego_path, transform=None):
        """
        Args:
            cover_path: Path to folder containing cover images (label 0)
            stego_path: Path to folder containing stego images (label 1)
            transform: Optional transform to apply to images
        """
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Expand home directory if present
        cover_path = os.path.expanduser(cover_path)
        stego_path = os.path.expanduser(stego_path)

        # Load cover images (label 0)
        cover_images = list(Path(cover_path).glob('*.pgm')) + \
                       list(Path(cover_path).glob('*.png')) + \
                       list(Path(cover_path).glob('*.jpg'))

        for img_path in cover_images:
            self.image_paths.append(str(img_path))
            self.labels.append(0)

        # Load stego images (label 1)
        stego_images = list(Path(stego_path).glob('*.pgm')) + \
                       list(Path(stego_path).glob('*.png')) + \
                       list(Path(stego_path).glob('*.jpg'))

        for img_path in stego_images:
            self.image_paths.append(str(img_path))
            self.labels.append(1)

        print(f"Loaded {len(cover_images)} cover images and {len(stego_images)} stego images")

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {cover_path} or {stego_path}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image and convert to RGB
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.FloatTensor([label])

# Training function
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = (outputs >= 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

    accuracy = accuracy_score(all_labels, all_preds)
    return total_loss / len(train_loader), accuracy

# Validation function
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = (outputs >= 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return total_loss / len(val_loader), accuracy, precision, recall, f1

def main(args):
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        # Remove normalization for steganography - subtle changes matter!
    ])

    val_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    # Create datasets
    print("Loading datasets...")
    print(f"Train cover path: {os.path.expanduser(args.cover_path)}")
    print(f"Train stego path: {os.path.expanduser(args.stego_path)}")
    print(f"Val cover path: {os.path.expanduser(args.valid_cover_path)}")
    print(f"Val stego path: {os.path.expanduser(args.valid_stego_path)}")

    train_dataset = StegoDataset(args.cover_path, args.stego_path, transform=train_transform)
    val_dataset = StegoDataset(args.valid_cover_path, args.valid_stego_path, transform=val_transform)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Initialize model
    model = StegoCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5, verbose=True)

    # Training loop
    print("\n" + "="*60)
    print("Training started...")
    print("="*60)

    best_val_acc = 0.0

    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*60}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_prec, val_rec, val_f1 = validate(model, val_loader, criterion, device)

        scheduler.step(val_acc)  # Schedule based on accuracy, not loss

        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"Val Precision: {val_prec:.4f} | Val Recall: {val_rec:.4f} | Val F1: {val_f1:.4f}")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'best_stego_classifier.pth')
            print(f"✓ Best model saved! (Val Acc: {val_acc:.4f})")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'checkpoint_epoch_{epoch+1}.pth')

    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Steganography Detection CNN')

    parser.add_argument(
        "--machine",
        type=str,
        default="server",
        choices=["local", "server"],
        help="Machine type: 'local' or 'server' (default: server)"
    )

    args, remaining = parser.parse_known_args()

    # Set paths based on machine type
    if args.machine == "local":
        default_cover_path = "/Users/dmitryhoma/Projects/phd_dissertation/state_3/INATNet/data/GBRASNET/BOSSbase-1.01-div/cover/train"
        default_stego_path = "/Users/dmitryhoma/Projects/phd_dissertation/state_3/INATNet/data/GBRASNET/BOSSbase-1.01-div/stego/S-UNIWARD/0.4bpp/stego/train"
        default_valid_cover_path = "/Users/dmitryhoma/Projects/phd_dissertation/state_3/INATNet/data/GBRASNET/BOSSbase-1.01-div/cover/val"
        default_valid_stego_path = "/Users/dmitryhoma/Projects/phd_dissertation/state_3/INATNet/data/GBRASNET/BOSSbase-1.01-div/stego/S-UNIWARD/0.4bpp/stego/val"
    else:  # server
        default_cover_path = "~/data/GBRASNET/BOSSbase-1.01-div/cover/train"
        default_stego_path = "~/data/GBRASNET/BOSSbase-1.01-div/stego/S-UNIWARD/0.4bpp/stego/train"
        default_valid_cover_path = "~/data/GBRASNET/BOSSbase-1.01-div/cover/val"
        default_valid_stego_path = "~/data/GBRASNET/BOSSbase-1.01-div/stego/S-UNIWARD/0.4bpp/stego/val"

    parser.add_argument(
        "--cover_path",
        default=default_cover_path,
        help="Path to training cover images"
    )
    parser.add_argument(
        "--stego_path",
        default=default_stego_path,
        help="Path to training stego images"
    )
    parser.add_argument(
        "--valid_cover_path",
        default=default_valid_cover_path,
        help="Path to validation cover images"
    )
    parser.add_argument(
        "--valid_stego_path",
        default=default_valid_stego_path,
        help="Path to validation stego images"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate (default: 0.0001)")
    parser.add_argument("--image_size", type=int, default=256, help="Image size (will be resized)")

    args = parser.parse_args()
    main(args)

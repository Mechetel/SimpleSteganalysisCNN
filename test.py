import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import argparse
from pathlib import Path
import os
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, confusion_matrix, classification_report)
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

        return image, torch.FloatTensor([label]), img_path

def test_model(model, test_loader, device):
    """Test the model and return predictions and metrics"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_paths = []

    print("\nRunning inference...")
    with torch.no_grad():
        for batch_idx, (images, labels, paths) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            preds = (outputs >= 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())
            all_paths.extend(paths)

            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed {batch_idx+1}/{len(test_loader)} batches")

    return np.array(all_preds), np.array(all_labels), np.array(all_probs), all_paths

def print_metrics(labels, preds):
    """Print comprehensive evaluation metrics"""
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    print("\n" + "="*60)
    print("Test Results")
    print("="*60)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print()

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    print("Confusion Matrix:")
    print(f"                Predicted")
    print(f"              Cover  Stego")
    print(f"Actual Cover   {cm[0][0]:5d}  {cm[0][1]:5d}")
    print(f"       Stego   {cm[1][0]:5d}  {cm[1][1]:5d}")
    print()

    # Classification report
    print("Classification Report:")
    print(classification_report(labels, preds, target_names=['Cover', 'Stego']))
    print("="*60)

def save_misclassified(labels, preds, paths, output_file='misclassified.txt'):
    """Save paths of misclassified images"""
    misclassified = []
    for i, (label, pred, path) in enumerate(zip(labels, preds, paths)):
        if label != pred:
            true_class = 'Cover' if label == 0 else 'Stego'
            pred_class = 'Cover' if pred == 0 else 'Stego'
            misclassified.append(f"{path} | True: {true_class} | Predicted: {pred_class}")

    if misclassified:
        with open(output_file, 'w') as f:
            f.write(f"Total misclassified: {len(misclassified)}\n")
            f.write("="*80 + "\n")
            for item in misclassified:
                f.write(item + "\n")
        print(f"\nMisclassified images saved to: {output_file}")
        print(f"Total misclassified: {len(misclassified)}")

def main(args):
    # Test transform (no augmentation)
    test_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Create test dataset
    print("Loading test dataset...")
    print(f"Test cover path: {os.path.expanduser(args.cover_path)}")
    print(f"Test stego path: {os.path.expanduser(args.stego_path)}")

    test_dataset = StegoDataset(args.cover_path, args.stego_path, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Load model
    print(f"Loading model from: {args.model_path}")
    model = StegoCNN().to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'val_acc' in checkpoint:
            print(f"Model validation accuracy: {checkpoint['val_acc']:.4f}")
    else:
        model.load_state_dict(checkpoint)

    # Test the model
    preds, labels, probs, paths = test_model(model, test_loader, device)

    # Print metrics
    print_metrics(labels, preds)

    # Save misclassified images
    if args.save_misclassified:
        save_misclassified(labels, preds, paths, args.misclassified_file)

    # Save predictions
    if args.save_predictions:
        np.savez(args.predictions_file,
                 predictions=preds,
                 labels=labels,
                 probabilities=probs,
                 paths=np.array(paths))
        print(f"\nPredictions saved to: {args.predictions_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Steganography Detection CNN')

    parser.add_argument(
        "--machine",
        type=str,
        default="server",
        choices=["local", "server"],
        help="Machine type: 'local' or 'server' (default: server)"
    )

    args, remaining = parser.parse_known_args()

    # Set paths based on machine type
    # FOR GBRASNET EXPERIMENTS
    # if args.machine == "local":
    #     default_cover_path = "/Users/dmitryhoma/Projects/phd_dissertation/state_3/INATNet/data/GBRASNET/BOSSbase-1.01-div/cover/val"
    #     default_stego_path = "/Users/dmitryhoma/Projects/phd_dissertation/state_3/INATNet/data/GBRASNET/BOSSbase-1.01-div/stego/S-UNIWARD/0.4bpp/stego/val"
    # else:  # server
    #     default_cover_path = "~/data/GBRASNET/BOSSbase-1.01-div/cover/val"
    #     default_stego_path = "~/data/GBRASNET/BOSSbase-1.01-div/stego/S-UNIWARD/0.4bpp/stego/val"

    # FOR BOSSBASE-1.01 EXPERIMENTS
    if args.machine == "local":
        default_cover_path = "/Users/dmitryhoma/Projects/phd_dissertation/state_3/SimpleSteganalysisCNN/data/boss_256_0.4_test/cover"
        default_stego_path = "/Users/dmitryhoma/Projects/phd_dissertation/state_3/SimpleSteganalysisCNN/data/boss_256_0.4_test/stego"
    else:  # server
        default_cover_path = "~/data/boss_256_0.4_test/cover"
        default_stego_path = "~/data/boss_256_0.4_test/stego"


    parser.add_argument(
        "--cover_path",
        default=default_cover_path,
        help="Path to test cover images"
    )
    parser.add_argument(
        "--stego_path",
        default=default_stego_path,
        help="Path to test stego images"
    )
    parser.add_argument(
        "--model_path",
        default="best_stego_classifier.pth",
        help="Path to trained model checkpoint"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--image_size", type=int, default=256, help="Image size (should match training)")
    parser.add_argument("--save_misclassified", action='store_true', help="Save misclassified images list")
    parser.add_argument("--misclassified_file", type=str, default="misclassified.txt",
                       help="File to save misclassified images")
    parser.add_argument("--save_predictions", action='store_true', help="Save all predictions to file")
    parser.add_argument("--predictions_file", type=str, default="predictions.npz",
                       help="File to save predictions")

    args = parser.parse_args()
    main(args)

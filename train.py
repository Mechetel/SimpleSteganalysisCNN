import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import argparse
import os
from model import StegoCNN, INATNet
from utils import get_data_loaders, calculate_metrics, save_checkpoint


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * images.size(0)

        # Get predictions (apply sigmoid for binary classification)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({'loss': loss.item()})

    epoch_loss = running_loss / len(train_loader.dataset)
    metrics = calculate_metrics(all_labels, all_preds)

    return epoch_loss, metrics


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item() * images.size(0)

            # Get predictions and probabilities
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    epoch_loss = running_loss / len(val_loader.dataset)
    metrics = calculate_metrics(all_labels, all_preds)

    return epoch_loss, metrics, all_labels, all_probs


def train(args):
    """Main training function"""

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Machine type: {args.machine}")
    print(f"Image size: {args.img_size}")

    # Create data loaders
    print("\n" + "="*70)
    print("Loading datasets...")
    print("="*70)
    print(f"Cover path:       {args.cover_path}")
    print(f"Stego path:       {args.stego_path}")
    print(f"Valid cover path: {args.valid_cover_path}")
    print(f"Valid stego path: {args.valid_stego_path}")

    train_loader, val_loader = get_data_loaders(
        args.cover_path,
        args.stego_path,
        args.valid_cover_path,
        args.valid_stego_path,
        args.batch_size,
        args.img_size,
        augment=True
    )

    print(f"\nTrain samples: {len(train_loader.dataset)}")
    if val_loader:
        print(f"Validation samples: {len(val_loader.dataset)}")

    # Initialize model
    # model = StegoCNN(pretrained=True, freeze_resnet=True).to(device)
    model = INATNet().to(device)
    print(f"\nModel initialized with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Training loop
    best_val_loss = float('inf')
    best_val_acc = 0.0

    print("\n" + "="*70)
    print("Starting training...")
    print("="*70 + "\n")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 70)

        # Train
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Print training metrics
        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Train Metrics - "
              f"Accuracy: {train_metrics['accuracy']:.4f}, "
              f"Precision: {train_metrics['precision']:.4f}, "
              f"Recall: {train_metrics['recall']:.4f}, "
              f"F1: {train_metrics['f1_score']:.4f}")

        # Validate if validation loader exists
        if val_loader:
            val_loss, val_metrics, _, _ = validate(
                model, val_loader, criterion, device
            )

            # Learning rate scheduler step
            scheduler.step(val_loss)

            print(f"\nVal Loss: {val_loss:.4f}")
            print(f"Val Metrics - "
                  f"Accuracy: {val_metrics['accuracy']:.4f}, "
                  f"Precision: {val_metrics['precision']:.4f}, "
                  f"Recall: {val_metrics['recall']:.4f}, "
                  f"F1: {val_metrics['f1_score']:.4f}")

            # Save best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_metrics['accuracy']
                model_path = os.path.join(args.output_dir, 'best_model.pth')
                save_checkpoint(model, optimizer, epoch, val_loss, model_path)
                print(f"✓ Best model updated! (Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc:.4f})")
        else:
            # If no validation set, save based on training loss
            if train_loss < best_val_loss:
                best_val_loss = train_loss
                model_path = os.path.join(args.output_dir, 'best_model.pth')
                save_checkpoint(model, optimizer, epoch, train_loss, model_path)
                print(f"✓ Best model updated! (Train Loss: {best_val_loss:.4f})")

    print("\n" + "="*70)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    if val_loader:
        print(f"Best validation accuracy: {best_val_acc:.4f}")
    print("="*70)

    # Save final model
    final_model_path = os.path.join(args.output_dir, 'final_model.pth')
    save_checkpoint(model, optimizer, args.epochs, train_loss, final_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Steganalysis Binary Classifier')

    # Machine type argument (parsed first)
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

    # FOR BOSSBASE-1.01 EXPERIMENTS (uncomment to use)
    # if args.machine == "local":
    #     default_cover_path = "/Users/dmitryhoma/Projects/phd_dissertation/state_3/SimpleSteganalysisCNN/data/boss_256_0.4/cover/train"
    #     default_stego_path = "/Users/dmitryhoma/Projects/phd_dissertation/state_3/SimpleSteganalysisCNN/data/boss_256_0.4/stego/train"
    #     default_valid_cover_path = "/Users/dmitryhoma/Projects/phd_dissertation/state_3/SimpleSteganalysisCNN/data/boss_256_0.4/cover/val"
    #     default_valid_stego_path = "/Users/dmitryhoma/Projects/phd_dissertation/state_3/SimpleSteganalysisCNN/data/boss_256_0.4/stego/val"
    # else:  # server
    #     default_cover_path = "~/data/boss_256_0.4/cover/train"
    #     default_stego_path = "~/data/boss_256_0.4/stego/train"
    #     default_valid_cover_path = "~/data/boss_256_0.4/cover/val"
    #     default_valid_stego_path = "~/data/boss_256_0.4/stego/val"

    # Data paths
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

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--img_size", type=int, default=256, help="Image size")
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                       help="Output directory for checkpoints")

    args = parser.parse_args()

    train(args)

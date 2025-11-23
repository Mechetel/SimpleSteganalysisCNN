import torch
import numpy as np
from PIL import Image
from pathlib import Path
import os

def check_images(cover_path, stego_path, num_samples=5):
    """Check if cover and stego images are actually different"""

    cover_path = os.path.expanduser(cover_path)
    stego_path = os.path.expanduser(stego_path)

    cover_images = list(Path(cover_path).glob('*.pgm'))[:num_samples]
    stego_images = list(Path(stego_path).glob('*.pgm'))[:num_samples]

    print("="*60)
    print("CHECKING IMAGE DIFFERENCES")
    print("="*60)

    for i, (cover_img_path, stego_img_path) in enumerate(zip(cover_images, stego_images)):
        cover = np.array(Image.open(cover_img_path))
        stego = np.array(Image.open(stego_img_path))

        print(f"\nPair {i+1}:")
        print(f"  Cover: {cover_img_path.name}")
        print(f"  Stego: {stego_img_path.name}")
        print(f"  Cover shape: {cover.shape}")
        print(f"  Stego shape: {stego.shape}")
        print(f"  Cover dtype: {cover.dtype}, range: [{cover.min()}, {cover.max()}]")
        print(f"  Stego dtype: {stego.dtype}, range: [{stego.min()}, {stego.max()}]")

        if cover.shape == stego.shape:
            diff = np.abs(cover.astype(float) - stego.astype(float))
            num_different = np.sum(diff > 0)
            percent_different = (num_different / diff.size) * 100

            print(f"  Pixels changed: {num_different} ({percent_different:.2f}%)")
            print(f"  Max difference: {diff.max()}")
            print(f"  Mean difference: {diff.mean():.4f}")

            if num_different == 0:
                print("  ⚠️  WARNING: Images are IDENTICAL!")
            elif percent_different < 1:
                print("  ✓ Subtle steganography detected")
            else:
                print("  ⚠️  Large differences - may not be stego pair")
        else:
            print("  ⚠️  ERROR: Image shapes don't match!")

def check_dataset_balance(cover_path, stego_path):
    """Check if dataset is balanced"""

    cover_path = os.path.expanduser(cover_path)
    stego_path = os.path.expanduser(stego_path)

    cover_count = len(list(Path(cover_path).glob('*.pgm')))
    stego_count = len(list(Path(stego_path).glob('*.pgm')))

    print("\n" + "="*60)
    print("DATASET BALANCE")
    print("="*60)
    print(f"Cover images: {cover_count}")
    print(f"Stego images: {stego_count}")
    print(f"Balance ratio: {min(cover_count, stego_count) / max(cover_count, stego_count):.2f}")

    if cover_count != stego_count:
        print("⚠️  WARNING: Dataset is imbalanced!")
    else:
        print("✓ Dataset is balanced")

def test_model_forward_pass():
    """Test if model can do a forward pass"""
    from model import StegoCNN

    print("\n" + "="*60)
    print("MODEL FORWARD PASS TEST")
    print("="*60)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = StegoCNN().to(device)

    # Create dummy batch
    dummy_input = torch.randn(4, 3, 256, 256).to(device)

    try:
        with torch.no_grad():
            output = model(dummy_input)

        print(f"✓ Forward pass successful")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output values: {output.squeeze().cpu().numpy()}")
        print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")

        if torch.all(output == 0.5):
            print("  ⚠️  WARNING: All outputs are exactly 0.5!")
        elif output.std() < 0.01:
            print("  ⚠️  WARNING: Very low variance in outputs - model may not be learning")

    except Exception as e:
        print(f"✗ Forward pass failed: {e}")

def check_gradient_flow():
    """Check if gradients are flowing"""
    from model import StegoCNN
    import torch.nn as nn

    print("\n" + "="*60)
    print("GRADIENT FLOW TEST")
    print("="*60)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = StegoCNN().to(device)
    criterion = nn.BCELoss()

    # Create dummy batch with different labels
    dummy_input = torch.randn(4, 3, 256, 256).to(device)
    dummy_labels = torch.FloatTensor([[0], [1], [0], [1]]).to(device)

    output = model(dummy_input)
    loss = criterion(output, dummy_labels)
    loss.backward()

    print(f"Loss: {loss.item():.4f}")

    # Check gradients in first and last layers
    first_layer = model.preprocessing[0]
    last_layer = model.fc_layers[-2]

    if first_layer.weight.grad is not None:
        first_grad_norm = first_layer.weight.grad.norm().item()
        print(f"First layer gradient norm: {first_grad_norm:.6f}")

        if first_grad_norm < 1e-7:
            print("  ⚠️  WARNING: Vanishing gradients in first layer!")
    else:
        print("  ⚠️  WARNING: No gradients in first layer!")

    if last_layer.weight.grad is not None:
        last_grad_norm = last_layer.weight.grad.norm().item()
        print(f"Last layer gradient norm: {last_grad_norm:.6f}")

        if last_grad_norm < 1e-7:
            print("  ⚠️  WARNING: Vanishing gradients in last layer!")
    else:
        print("  ⚠️  WARNING: No gradients in last layer!")

if __name__ == "__main__":
    # Use your paths
    train_cover = "~/data/GBRASNET/BOSSbase-1.01-div/cover/train"
    train_stego = "~/data/GBRASNET/BOSSbase-1.01-div/stego/S-UNIWARD/0.4bpp/stego/train"

    print("\n" + "="*60)
    print("STEGANOGRAPHY DATASET DIAGNOSTICS")
    print("="*60)

    check_images(train_cover, train_stego, num_samples=5)
    check_dataset_balance(train_cover, train_stego)
    test_model_forward_pass()
    check_gradient_flow()

    print("\n" + "="*60)
    print("DIAGNOSTICS COMPLETE")
    print("="*60)

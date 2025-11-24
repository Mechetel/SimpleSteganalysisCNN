import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm
import argparse
import os
from model import StegoCNN, INATNet
from utils import get_test_loader, calculate_metrics


def test_model(args):
    """
    Test the model and generate ROC curve and AUC
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Machine type: {args.machine}")

    # Load test data
    print("\n" + "="*70)
    print("Loading test dataset...")
    print("="*70)
    print(f"Cover path: {args.cover_path}")
    print(f"Stego path: {args.stego_path}")

    test_loader = get_test_loader(
        args.cover_path,
        args.stego_path,
        args.batch_size,
        args.img_size
    )
    print(f"Test samples: {len(test_loader.dataset)}")

    # Load model
    print(f"\nLoading model from {args.model_path}...")
    # model = StegoCNN(pretrained=True, freeze_resnet=True).to(device)
    model = INATNet().to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully!")

    # Evaluate
    print("\n" + "="*70)
    print("Evaluating model...")
    print("="*70)
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images).squeeze()

            # Get predictions and probabilities
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate metrics
    print("\n" + "="*70)
    print("Test Results")
    print("="*70)

    metrics = calculate_metrics(all_labels, all_preds)
    print(f"\nAccuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")

    # Classification report
    print("\n" + "="*70)
    print("Classification Report")
    print("="*70)
    print(classification_report(all_labels, all_preds,
                                target_names=['Cover (0)', 'Stego (1)'],
                                digits=4))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(f"{'':>12} {'Predicted Cover':>18} {'Predicted Stego':>18}")
    print(f"{'True Cover':<12} {cm[0,0]:>18} {cm[0,1]:>18}")
    print(f"{'True Stego':<12} {cm[1,0]:>18} {cm[1,1]:>18}")

    # ROC Curve and AUC
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    print("\n" + "="*70)
    print(f"AUC (Area Under ROC Curve): {roc_auc:.4f} ({roc_auc*100:.2f}%)")
    print("="*70)

    # Find optimal threshold using Youden's J statistic
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]

    print(f"\nOptimal Threshold (Youden's J): {optimal_threshold:.4f}")
    print(f"At this threshold:")
    print(f"  - True Positive Rate (Sensitivity):  {tpr[optimal_idx]:.4f}")
    print(f"  - False Positive Rate:                {fpr[optimal_idx]:.4f}")
    print(f"  - True Negative Rate (Specificity):  {1-fpr[optimal_idx]:.4f}")

    # Calculate metrics at optimal threshold
    preds_optimal = (all_probs >= optimal_threshold).astype(int)
    metrics_optimal = calculate_metrics(all_labels, preds_optimal)
    print(f"  - Accuracy at optimal threshold:      {metrics_optimal['accuracy']:.4f}")

    # Plot ROC Curve and other visualizations
    plt.figure(figsize=(18, 5))

    # Subplot 1: ROC Curve
    plt.subplot(1, 3, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random Classifier (AUC = 0.5)')
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', s=100,
                zorder=5, label=f'Optimal Threshold = {optimal_threshold:.4f}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    # Subplot 2: Confusion Matrix
    plt.subplot(1, 3, 2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Cover', 'Stego'],
                yticklabels=['Cover', 'Stego'],
                cbar_kws={'label': 'Count'})
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')

    # Add percentages
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            plt.text(j+0.5, i+0.7, f'({cm[i,j]/total*100:.1f}%)',
                    ha='center', va='center', color='gray', fontsize=9)

    # Subplot 3: Probability Distribution
    plt.subplot(1, 3, 3)
    plt.hist(all_probs[all_labels == 0], bins=50, alpha=0.6,
             label='Cover (True Negative)', color='blue', edgecolor='black')
    plt.hist(all_probs[all_labels == 1], bins=50, alpha=0.6,
             label='Stego (True Positive)', color='red', edgecolor='black')
    plt.axvline(x=0.5, color='green', linestyle='--', linewidth=2,
                label='Default Threshold (0.5)')
    plt.axvline(x=optimal_threshold, color='orange', linestyle='--', linewidth=2,
                label=f'Optimal Threshold ({optimal_threshold:.3f})')
    plt.xlabel('Predicted Probability', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3, axis='y')

    plt.tight_layout()

    # Save plot
    output_path = os.path.join(args.output_dir, 'test_results.png')
    os.makedirs(args.output_dir, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plots saved to '{output_path}'")

    if args.show_plot:
        plt.show()

    # Save results to text file
    results_path = os.path.join(args.output_dir, 'test_results.txt')
    with open(results_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("Steganalysis Test Results\n")
        f.write("="*70 + "\n\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Cover path: {args.cover_path}\n")
        f.write(f"Stego path: {args.stego_path}\n\n")
        f.write(f"Test samples: {len(test_loader.dataset)}\n\n")
        f.write(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall:    {metrics['recall']:.4f}\n")
        f.write(f"F1 Score:  {metrics['f1_score']:.4f}\n")
        f.write(f"AUC:       {roc_auc:.4f} ({roc_auc*100:.2f}%)\n\n")
        f.write(f"Optimal Threshold: {optimal_threshold:.4f}\n")
        f.write(f"Accuracy at optimal threshold: {metrics_optimal['accuracy']:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"{'':>12} {'Predicted Cover':>18} {'Predicted Stego':>18}\n")
        f.write(f"{'True Cover':<12} {cm[0,0]:>18} {cm[0,1]:>18}\n")
        f.write(f"{'True Stego':<12} {cm[1,0]:>18} {cm[1,1]:>18}\n")

    print(f"✓ Results saved to '{results_path}'")

    return {
        'metrics': metrics,
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr,
        'optimal_threshold': optimal_threshold
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Steganalysis Binary Classifier')

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
        default_cover_path = "/Users/dmitryhoma/Projects/phd_dissertation/state_3/INATNet/data/GBRASNET/BOSSbase-1.01-div/cover/val"
        default_stego_path = "/Users/dmitryhoma/Projects/phd_dissertation/state_3/INATNet/data/GBRASNET/BOSSbase-1.01-div/stego/S-UNIWARD/0.4bpp/stego/val"
    else:  # server
        default_cover_path = "~/data/GBRASNET/BOSSbase-1.01-div/cover/val"
        default_stego_path = "~/data/GBRASNET/BOSSbase-1.01-div/stego/S-UNIWARD/0.4bpp/stego/val"

    # FOR BOSSBASE-1.01 EXPERIMENTS (uncomment to use)
    # if args.machine == "local":
    #     default_cover_path = "/Users/dmitryhoma/Projects/phd_dissertation/state_3/SimpleSteganalysisCNN/data/boss_256_0.4_test/cover"
    #     default_stego_path = "/Users/dmitryhoma/Projects/phd_dissertation/state_3/SimpleSteganalysisCNN/data/boss_256_0.4_test/stego"
    # else:  # server
    #     default_cover_path = "~/data/boss_256_0.4_test/cover"
    #     default_stego_path = "~/data/boss_256_0.4_test/stego"

    # Data paths
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

    # Model and testing parameters
    parser.add_argument("--model_path", type=str, default="./checkpoints/best_model.pth",
                       help="Path to trained model checkpoint")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--img_size", type=int, default=256, help="Image size")
    parser.add_argument("--output_dir", type=str, default="./test_results",
                       help="Output directory for results")
    parser.add_argument("--show_plot", action="store_true",
                       help="Show plot after testing")

    args = parser.parse_args()

    results = test_model(args)

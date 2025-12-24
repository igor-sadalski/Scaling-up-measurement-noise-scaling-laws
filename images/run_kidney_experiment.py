# run_kidney_experiment.py
import os
import sys
from pathlib import Path
from typing import Optional, Callable, Tuple
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from latentmi import lmi, ksg
import pandas as pd
import medmnist
from medmnist import INFO


def add_gauss(x: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    """Add Gaussian noise to image."""
    return x + torch.randn_like(x) * sigma


def pixelate(x: torch.Tensor, scale: int = 4) -> torch.Tensor:
    """Pixelate image by downsampling and upsampling."""
    h, w = x.shape[-2:]
    small = F.avg_pool2d(x, kernel_size=scale, stride=scale)
    return small#F.interpolate(small, size=(h, w), mode='nearest')
# def pixelate(x: torch.Tensor, scale: int = 4) -> torch.Tensor:
#     """Downsample image by averaging patches, matching downsample_resolution behavior."""
#     # x shape: (batch, channels, height, width)
#     b, c, h, w = x.shape
#     new_h, new_w = h // scale, w // scale
    
#     downsampled = torch.zeros((b, c, new_h, new_w), device=x.device, dtype=x.dtype)
    
#     for i in range(new_h):
#         for j in range(new_w):
#             # Extract patch
#             patch = x[:, :, i * scale:(i + 1) * scale, j * scale:(j + 1) * scale]
#             # Average across height and width dimensions
#             avg_patch = torch.mean(patch, dim=(2, 3))
#             downsampled[:, :, i, j] = avg_patch
    
#     return downsampled

def create_model(num_classes: int, device: torch.device) -> nn.Module:
    """Create MobileNetV3-small model adapted for 3-channel input."""
    model = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
    
    # MobileNetV3 already expects 3 channels, but we'll modify classifier
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    
    return model.to(device)


def train_model(model: nn.Module, train_loader: DataLoader, noise_fn: Optional[Callable],
                device: torch.device, n_epochs: int = 50, lr: float = 1e-3) -> None:
    """Train the model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in tqdm(range(n_epochs), desc="Epochs", leave=True):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False):
            x, y = x.to(device), y.squeeze().long().to(device)
            
            # Apply noise if specified
            # if noise_fn:
            #     x = noise_fn(x)
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
        
        avg_loss = epoch_loss / len(train_loader)
        accuracy = 100. * correct / total
        print(f"Epoch {epoch+1}/{n_epochs} | Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%")


def compute_mi(model: nn.Module, data_loader: DataLoader, noise_fn: Optional[Callable],
               device: torch.device, num_classes: int) -> Tuple[float, float, dict, dict]:
    """Compute mutual information between representations and labels.
    
    Returns:
        mi_score: Continuous MI from representations
        discrete_mi: Overall discrete MI
        ova_mis_discrete: Dictionary mapping class indices to discrete one-vs-all MIs
        ova_mis_continuous: Dictionary mapping class indices to continuous one-vs-all MIs
    """
    model.eval()
    
    representations = []
    targets = []
    y_hats = []
    
    # Create hook to extract features
    features = {}
    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook
    
    # Register hook on last layer before classifier
    model.classifier[0].register_forward_hook(get_features('last_layer'))
    
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.squeeze().long()
            
            # Apply noise if specified
            # if noise_fn:
            #     x = noise_fn(x)
            
            y_hat = model(x)
            representations.append(features['last_layer'].cpu())
            targets.append(y)
            y_hats.append(y_hat.cpu())
    
    # Convert to numpy arrays
    X = torch.cat(representations).numpy()
    Y = torch.cat(targets).numpy()
    Y_onehot = np.eye(num_classes)[Y]
    Y_hats_onehot = torch.cat(y_hats).numpy()
    Y_hats = np.argmax(Y_hats_onehot, axis=1)
    
    print(f"Representations shape: {X.shape}, Labels shape: {Y.shape}, Predictions shape: {Y_hats.shape}")
    
    # Calculate MI using latentmi
    pmis, _, _ = lmi.estimate(X, Y_onehot, validation_split=0.3, batch_size=512, 
                             epochs=50, quiet=False)
    mi_score = np.nanmean(pmis)
    
    print(f"MI score (representation): {mi_score:.4f}")

    discrete_mi = ksg.midd(Y, Y_hats)
    print(f"MI score (discrete): {discrete_mi:.4f}")
    
    # Compute one-vs-all MIs with balanced resampling
    print("\nComputing one-vs-all MIs (discrete and continuous)...")
    ova_mis_discrete = {}
    ova_mis_continuous = {}
    
    for class_idx in range(num_classes):
        # Create binary labels (1 for current class, 0 for all others)
        Y_binary = (Y == class_idx).astype(int)
        Y_hats_binary = (Y_hats == class_idx).astype(int)
        
        # Count samples in each binary class
        n_positive = np.sum(Y_binary)
        n_negative = len(Y_binary) - n_positive
        
        # Balanced resampling: take min of the two classes and sample equally
        n_samples = min(n_positive, n_negative)
        
        if n_samples == 0:
            print(f"  Class_{class_idx}: skipped (no samples)")
            ova_mis_discrete[f"Class_{class_idx}"] = np.nan
            ova_mis_continuous[f"Class_{class_idx}"] = np.nan
            continue
        
        positive_indices = np.where(Y_binary == 1)[0]
        negative_indices = np.where(Y_binary == 0)[0]
        
        # Randomly sample n_samples from each class
        np.random.seed(42)  
        sampled_positive = np.random.choice(positive_indices, size=n_samples, replace=True)
        sampled_negative = np.random.choice(negative_indices, size=n_samples, replace=True)
        
        # Combine and shuffle
        balanced_indices = np.concatenate([sampled_positive, sampled_negative])
        np.random.shuffle(balanced_indices)
        
        # Get balanced labels, predictions, and representations
        Y_binary_balanced = Y_binary[balanced_indices]
        Y_hats_binary_balanced = Y_hats_binary[balanced_indices]
        X_balanced = X[balanced_indices]
        
        # Compute discrete MI for this one-vs-all binary classification
        ova_mi_discrete = ksg.midd(Y_binary_balanced.reshape(-1, 1), Y_hats_binary_balanced.reshape(-1, 1))
        ova_mis_discrete[f"Class_{class_idx}"] = ova_mi_discrete
        
        # Compute continuous MI using representations
        Y_binary_onehot = np.eye(2)[Y_binary_balanced]
        pmis_ova, _, _ = lmi.estimate(X_balanced, Y_binary_onehot, validation_split=0.3, 
                                      batch_size=512, epochs=50, quiet=True)
        ova_mi_continuous = np.nanmean(pmis_ova)
        ova_mis_continuous[f"Class_{class_idx}"] = ova_mi_continuous
        
        print(f"  Class_{class_idx} (balanced {n_samples} per class): discrete={ova_mi_discrete:.4f} bits, continuous={ova_mi_continuous:.4f} bits")
    
    return mi_score, discrete_mi, ova_mis_discrete, ova_mis_continuous


def run_experiment(dataset_name: str, output_dir: str, noise_fn: Optional[Callable],
                   tag: str, device: torch.device, n_epochs: int = 50,
                   size: int = 224, download: bool = True) -> Tuple[float, float, dict]:
    """Run a single noise experiment."""
    print(f"\n{'='*60}")
    print(f"Starting experiment: {tag}")
    print(f"{'='*60}")
    
    # Get dataset info
    info = INFO[dataset_name]
    num_classes = len(info['label'])
    DataClass = getattr(medmnist, info['python_class'])
    
    print(f"Dataset: {dataset_name}")
    print(f"Number of classes: {num_classes}")
    print(f"Labels: {info['label']}")
    
    # Prepare transforms - convert grayscale to RGB by repeating channel
    # tfm = transforms.Compose([
    #     transforms.Resize((size, size)),
    #     transforms.ToTensor(),
    #     transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
    # ])
    if noise_fn:
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
            transforms.Lambda(noise_fn),
            transforms.Resize((size, size)),
        ])
    else:
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
            transforms.Resize((size, size)),
        ])
    
    # Load datasets
    train_ds = DataClass(split='train', transform=tfm, download=download, size=size)
    test_ds = DataClass(split='test', transform=tfm, download=download, size=size)
    
    train_dl = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=4)
    test_dl = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=4)
    
    print(f'Dataset split: {len(train_ds)} train, {len(test_ds)} test')
    
    # Create and train model
    model = create_model(num_classes, device)
    
    train_model(model, train_dl, noise_fn, device, n_epochs=n_epochs)
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"model_{tag}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")
    
    # Compute MI
    mi_score, discrete_mi, ova_mis, ova_mis_continuous = compute_mi(model, test_dl, noise_fn, device, num_classes)
    
    return mi_score, discrete_mi, ova_mis, ova_mis_continuous


def main():
    parser = argparse.ArgumentParser(description='Run kidney pilot experiment with specific noise level')
    parser.add_argument('--noise-type', type=str, required=True, choices=['clean', 'gauss', 'pix'],
                        help='Type of noise to apply')
    parser.add_argument('--noise-param', type=float, default=None,
                        help='Noise parameter (sigma for gauss, scale for pix)')
    parser.add_argument('--dataset', type=str, default='tissuemnist',
                        help='Dataset name (default: tissuemnist)')
    parser.add_argument('--output-dir', type=str, default='tissuemnist_models',
                        help='Output directory for models and results')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--size', type=int, default=224,
                        help='Image size')
    parser.add_argument('--no-download', action='store_true',
                        help='Skip downloading dataset (must already exist)')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create noise function and tag based on arguments
    if args.noise_type == 'clean':
        noise_fn = None
        tag = 'clean'
    elif args.noise_type == 'gauss':
        if args.noise_param is None:
            raise ValueError("--noise-param required for gaussian noise")
        sigma = args.noise_param
        noise_fn = lambda x: add_gauss(x, sigma)
        tag = f"gauss_{sigma:.6f}"
    elif args.noise_type == 'pix':
        if args.noise_param is None:
            raise ValueError("--noise-param required for pixelation")
        scale = int(args.noise_param)
        noise_fn = lambda x: pixelate(x, scale)
        tag = f"pix_{scale}x"
    
    try:
        mi_score, discrete_mi, ova_mis, ova_mis_continuous = run_experiment(
            args.dataset,
            args.output_dir,
            noise_fn,
            tag,
            device,
            n_epochs=args.epochs,
            size=args.size,
            download=not args.no_download
        )
        
        # Save individual result
        result = {
            'experiment': tag,
            'mi_score': mi_score,
            'discrete_mi': discrete_mi,
            **{f'ova_mi_{k}': v for k, v in ova_mis.items()},
            **{f'ova_mi_continuous_{k}': v for k, v in ova_mis_continuous.items()}
        }
        
        result_path = os.path.join(args.output_dir, f'result_{tag}.csv')
        pd.DataFrame([result]).to_csv(result_path, index=False)
        print(f"\nResult saved to {result_path}")
        
    except Exception as e:
        print(f"Error in experiment {tag}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
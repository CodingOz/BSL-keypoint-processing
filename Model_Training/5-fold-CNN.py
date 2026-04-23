"""
1D CNN with Stratified K-Fold Cross-Validation for BSL Sign Classification.
Architecture: 1D convolutions along the temporal axis (10 frames).
Features are treated as input channels, time as the spatial dimension.
"""
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import copy
import os


class BSLDataset(Dataset):
    """BSL sign language dataset with optional on-the-fly augmentation."""

    def __init__(self, features, labels, augment=False, noise_std=0.03,
                 time_shift_max=1, scale_range=(0.95, 1.05)):
        """
        takes:
            features: np.ndarray of shape (N, num_features, num_frames)
                      already scaled and transposed.
            labels:   np.ndarray of int class indices.
            augment:  whether to apply random augmentation (training only).
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.augment = augment
        self.noise_std = noise_std
        self.time_shift_max = time_shift_max
        self.scale_range = scale_range

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.features[idx].clone()
        y = self.labels[idx]

        if self.augment:
            # additive Gaussian noise
            x = x + torch.randn_like(x) * self.noise_std

            # Random temporal shift
            shift = np.random.randint(-self.time_shift_max,
                                      self.time_shift_max + 1)
            if shift != 0:
                x = torch.roll(x, shifts=shift, dims=-1)

            # random per-frame scaling
            lo, hi = self.scale_range
            scale = torch.empty(1, x.shape[-1]).uniform_(lo, hi)
            x = x * scale

        return x, y


class BSLConv1DNet(nn.Module):
    """
    Small 1D CNN for temporal sign classification.

    Input shape:  (batch, in_channels, 10), features as channels, frames as time
    Output shape: (batch, num_classes)

    Architecture:
        Conv1D(in_ch, 32, k=3, pad=1) to BN to ReLU to Dropout
        Conv1D(32, 64, k=3, pad=1) to BN to ReLU to Dropout
        Conv1D(64, 64, k=3, pad=1) to BN to ReLU to Dropout
        Global Average Pooling to collapses time dim
        Linear(64, num_classes)
    """

    def __init__(self, in_channels, num_classes, dropout=0.35):
        super().__init__()

        self.conv_block = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            # Block 2
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            # Block 3
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Global average pooling collapses time
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.conv_block(x)       # (batch, 64, time)
        x = self.gap(x).squeeze(-1)  # (batch, 64)
        x = self.classifier(x)       # (batch, num_classes)
        return x


class EarlyStopping:
    """Stop training when validation loss doesn't improve for `patience` epochs."""

    def __init__(self, patience=15, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.best_model = None
        self.counter = 0

    def step(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience  # stop if patience exceeded


def train_one_epoch(model, loader, criterion, optimiser, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimiser.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimiser.step()

        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)

        total_loss += loss.item() * y.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    return total_loss / total, correct / \
        total, np.array(all_preds), np.array(all_labels)


def load_corpus(path):
    """Load corpus_data.json and return features array + label array."""
    with open(path, 'r') as f:
        data = json.load(f)

    features = []  # will be (N, frames, feats_per_frame)
    labels = []

    for sample in data:
        features.append(sample['features'])
        labels.append(sample['sign'])

    features = np.array(features, dtype=np.float32)  # (N, frames, feats)
    return features, np.array(labels)


def prepare_fold(X_all, y_encoded, train_idx, val_idx):
    """
    Scale features per-fold (fit on train, transform both).
    Transpose to (N, channels, time) for Conv1D.
    """
    # X_all shape: (N, frames, features_per_frame)
    n_frames = X_all.shape[1]
    n_feats = X_all.shape[2]

    # Flatten frames × features for scaling, then reshape back
    X_train_flat = X_all[train_idx].reshape(len(train_idx), -1)
    X_val_flat = X_all[val_idx].reshape(len(val_idx), -1)

    scaler = StandardScaler()
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_val_flat = scaler.transform(X_val_flat)

    X_train = X_train_flat.reshape(len(train_idx), n_frames, n_feats)
    X_val = X_val_flat.reshape(len(val_idx), n_frames, n_feats)

    # Transpose to (N, features, frames) and features become Conv1D channels
    X_train = X_train.transpose(0, 2, 1)  # (N, feats, frames)
    X_val = X_val.transpose(0, 2, 1)

    return X_train, X_val


def run_kfold(corpus_path, n_folds=5, n_epochs=150, batch_size=16,
              lr=1e-3, weight_decay=1e-4, dropout=0.35, patience=15,
              augment=True, device=None):
    """Run stratified k-fold cross-validation."""

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Load data
    X_all, y_raw = load_corpus(corpus_path)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)
    class_names = le.classes_

    n_samples, n_frames, n_feats = X_all.shape
    n_classes = len(class_names)

    print(f"Corpus: {os.path.basename(os.path.dirname(corpus_path))}")
    print(f"Samples: {n_samples}  |  Frames: {n_frames}  |  "
          f"Features/frame: {n_feats}  |  Classes: {n_classes}")
    print(
        f"Class distribution: {dict(zip(*np.unique(y_raw, return_counts=True)))}")
    print(f"Model input shape: (batch, {n_feats}, {n_frames})")
    print(f"Augmentation: {'ON' if augment else 'OFF'}")
    print("=" * 70)

    # K-Fold setup
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_results = []
    all_val_preds = np.zeros(n_samples, dtype=int)
    all_val_labels = np.zeros(n_samples, dtype=int)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, y_encoded)):
        print(f"\n{'─' * 30} Fold {fold + 1}/{n_folds} {'─' * 30}")
        print(
            f"Train: {len(train_idx)} samples  |  Val: {len(val_idx)} samples")

        # prepare data
        X_train, X_val = prepare_fold(X_all, y_encoded, train_idx, val_idx)
        y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

        train_ds = BSLDataset(X_train, y_train, augment=augment)
        val_ds = BSLDataset(X_val, y_val, augment=False)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=len(val_idx))

        # build model
        model = BSLConv1DNet(in_channels=n_feats, num_classes=n_classes,
                             dropout=dropout).to(device)
        criterion = nn.CrossEntropyLoss()
        optimiser = torch.optim.Adam(model.parameters(), lr=lr,
                                     weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode='min', factor=0.5, patience=7
        )
        stopper = EarlyStopping(patience=patience)

        # Training loop
        best_val_acc = 0.0
        for epoch in range(1, n_epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimiser, device
            )
            val_loss, val_acc, preds, labels = evaluate(
                model, val_loader, criterion, device
            )
            scheduler.step(val_loss)

            if val_acc > best_val_acc:
                best_val_acc = val_acc

            if epoch % 25 == 0 or epoch == 1:
                print(
                    f"  Epoch {epoch:3d}  "
                    f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
                    f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")

            if stopper.step(val_loss, model):
                print(
                    f"  Early stop at epoch {epoch}  (best val_loss={stopper.best_loss:.4f})")
                break

        # Evaluate best model for this fold
        model.load_state_dict(stopper.best_model)
        _, final_acc, final_preds, final_labels = evaluate(
            model, val_loader, criterion, device
        )

        all_val_preds[val_idx] = final_preds
        all_val_labels[val_idx] = final_labels

        fold_results.append({
            'fold': fold + 1,
            'val_acc': final_acc,
            'stopped_epoch': epoch,
            'best_val_loss': stopper.best_loss,
        })

        print(f"\n  Fold {fold + 1} result: val_acc = {final_acc:.3f}")

    print("\n" + "=" * 70)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 70)

    accs = [r['val_acc'] for r in fold_results]
    for r in fold_results:
        print(f"  Fold {r['fold']}: acc={r['val_acc']:.3f}  "
              f"stopped_epoch={r['stopped_epoch']}  "
              f"best_val_loss={r['best_val_loss']:.4f}")

    print(f"\n  Mean accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"  Min: {np.min(accs):.3f}  Max: {np.max(accs):.3f}")

    # Per-class report (aggregated across all folds)
    print(f"\n{'─' * 70}")
    print("PER-CLASS REPORT (aggregated across all folds)")
    print(f"{'─' * 70}")
    print(classification_report(all_val_labels, all_val_preds,
                                target_names=class_names, digits=3))

    # Confusion matrix
    cm = confusion_matrix(all_val_labels, all_val_preds)
    print("Confusion Matrix:")
    # Header
    header = "      " + "  ".join(f"{c:>3}" for c in class_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:3d}" for v in row)
        print(f"  {class_names[i]:>3}  {row_str}")

    return fold_results, all_val_preds, all_val_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='1D CNN k-fold CV for BSL sign classification')
    parser.add_argument('--corpus', type=str, required=True,
                        help='Path to corpus_data.json')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.35)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--no_augment', action='store_true')

    args = parser.parse_args()

    run_kfold(
        corpus_path=args.corpus,
        n_folds=args.folds,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        patience=args.patience,
        augment=not args.no_augment,
    )
